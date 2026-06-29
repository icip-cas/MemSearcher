# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import os
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from verl.utils.dataset.template import re_call_template_sys
from recurrent.utils import TokenTemplate, pad_tensor_list_to_length, create_attention_mask, create_position_ids
import copy

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
                train_tp = kwargs.get('train_tp', None)
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.token_message_template = TokenTemplate(re_call_template_sys, tokenizer)
        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length + self.token_message_template.length + self.config.memory_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            max_num_batched_tokens = max_model_len
            # raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
            #                  please increase max_num_batched_tokens or disable chunked prefill')

        trust_remote_code = kwargs.get('trust_remote_code', False)
        load_format = 'dummy' if config.load_format.startswith('dummy') else config.load_format

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=int(os.getenv("RANK", "0")) // tensor_parallel_size,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id,
                                             max_length=self.config.response_length).to(idx.device)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                                self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response,
                                                    eos_token=eos_token_id,
                                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

import re
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from verl.utils.torch_functional import pad_sequence_to_length

def get_position_ids(attention_mask: torch.Tensor):
    # 创建一个和 attention_mask 同形状的 tensor，每一行为按行累加但仅在 mask==1 时递增
    cumsum = torch.cumsum(attention_mask, dim=1)
    # 把没有被 mask 的位置（即为 0 的位置）置为 0，其他位置减去1使得从0开始
    position_ids = (cumsum - 1) * attention_mask
    return position_ids

class vLLMRolloutWithTool(vLLMRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        self.gen_str = "\n<|im_start|>assistant\n"
        self.gen_ids = self.tokenizer.encode(self.gen_str)
        self.max_input_length = config.prompt_length + self.token_message_template.length + self.config.memory_length
        self.NO_MEMORY_TOKENS = tokenizer.encode("", add_special_tokens=False)
    
    def format_tool_call(self, tool_call_str: str):
        """Convert JSON function call description to Python executable code string."""
        try:
            call_json = json.loads(tool_call_str)
            func_name = call_json['name']
            arguments = call_json.get('arguments', {})
            
            args_str = ', '.join(f"{k}={repr(v)}" for k, v in arguments.items())
            return f"{func_name}({args_str})"
        except Exception as e:
            return f"Parse tool call failed: {e}"

    def validate_tool_calls(self, output_str):
        start_tags = re.findall(r'<tool_call>', output_str)
        end_tags = re.findall(r'</tool_call>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(r'<tool_call>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</tool_call>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    def extract_tool_calls(self, output_str):
        if not self.validate_tool_calls(output_str):
            return []

        try:
            pattern = r'<tool_call>((?:(?!</tool_call>).)*)</tool_call>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []

    def validate_memory(self, output_str):
        start_tags = re.findall(r'<memory>', output_str)
        end_tags = re.findall(r'</memory>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(r'<memory>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</memory>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    def extract_memory(self, output_str):
        if not self.validate_memory(output_str):
            return ""
        
        try:
            pattern = r'<memory>((?:(?!</memory>).)*)</memory>'
            match = re.search(pattern, output_str, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return ""
        except Exception as e:
            return ""
    
    def batch_execute(self, env_list: List[str], tool_calls_list: List[List[str]]):
        def exe_tool_call(env, call):
            url = f'{self.config.sandbox_url}/execute'

            call_str = self.format_tool_call(call)
            if call_str.startswith("Parse tool call failed"):
                return call_str
            
            try:
                data = {
                    'env': env,
                    'call': call_str
                }                
                response = requests.post(url, json=data, timeout=10)
                if response.status_code != 200:
                    return f"error: {response.status_code}"
                response = response.json()
                ret_str = ''
                if response['result']:
                    ret_str += f'result: \n{response["result"]}\n'
                if response['output']:
                    ret_str += f'output: \n{response["output"]}\n'
                if response['error']:
                    ret_str += f'error: \n{response["error"]}\n'
                return ret_str.strip()
            except requests.exceptions.Timeout:
                return "error: execution timed out"
            except Exception as e:
                return str(e)

        def wikipedia_search(tool_call):
            # Retriever endpoint, configurable via the SEARCH_URL env var (propagated to
            # rollout workers in main_ppo). Defaults to a co-located retriever on localhost.
            _base = os.environ.get("SEARCH_URL", "http://127.0.0.1:8000").rstrip("/")
            url = _base if _base.endswith("/search") else _base + "/search"
            try:
                data = json.loads(tool_call)["arguments"]
            except Exception as e:
                return "Parse tool call failed"
            
            try:
                query = data.get('query', '')
                if  query== '':
                    return 'invalid query'
            except Exception as e:
                return "Parse query failed"
            
            # data = {'query': query, 'top_n': top_n}
            try:
                response = requests.post(url, json=data, timeout=10)
                if response.status_code != 200:
                    return f"error: {response.status_code}"
                retrieval_text = ''
                for line in response.json():
                    retrieval_text += f"{line['contents']}\\n\\n"
                retrieval_text = retrieval_text.strip()
                
                return retrieval_text
            except requests.exceptions.Timeout:
                return "error: execution timed out"
            except Exception as e:
                return str(e)

        # flatten all tasks
        all_tasks = []
        task_indices = []
        for env_idx, (env, tool_calls) in enumerate(zip(env_list, tool_calls_list)):
            for call_idx, tool_call in enumerate(tool_calls):
                all_tasks.append((env, tool_call))
                task_indices.append((env_idx, call_idx))

        # parallel execute all tasks
        all_results = [None] * len(all_tasks)
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_index = {executor.submit(wikipedia_search, tool_call): i 
                        for i, (_, tool_call) in enumerate(all_tasks)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                all_results[index] = future.result()

        # reorganize results to original structure
        results_list = [[None for _ in range(len(tool_calls_list[i]))] for i, _ in enumerate(env_list)]
        for (env_idx, call_idx), result in zip(task_indices, all_results):
            results_list[env_idx][call_idx] = result

        return results_list

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:        
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        # ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        # attention_mask = prompts.batch['attention_mask']
        # position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        # eos_token_id = prompts.meta_info['eos_token_id']

        # batch_size = ori_input_ids.size(0)
        batch_size = len(prompts)

        # idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        # for i in range(batch_size):
        #     idx_list.append(_pre_process_inputs(self.pad_token_id, ori_input_ids[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        with self.update_sampling_params(**kwargs):
            # prepare n copies for each input
            # curr_inputs = []
            # for input_ids in idx_list:
            #     # for _ in range(self.sampling_params.n):
            #     curr_inputs.append(input_ids.copy())
            # init_inputs = [ids.copy() for ids in curr_inputs]

            # if there are envs, prepare n copies for each env
            env_list = None
            if 'env' in prompts.non_tensor_batch:
                env_list = []
                for env in prompts.non_tensor_batch['env']:
                    for _ in range(self.sampling_params.n):
                        env_list.append(env)

            # track the status of each input
            # curr_max_tokens = [self.sampling_params.max_tokens] * len(curr_inputs)
            active_indices = list(range(batch_size))

            # collect the result mask of each rollout, 1 for non-result, 0 for tool call result or pad
            # result_mask_list = [[] for _ in range(batch_size)]

            return_input_ids = []
            return_seq = []
            return_result_mask_list = []
            final_mask = []
            sample_index = []
            # memory = np.empty(batch_size, dtype=object)
            memory = [[] for _ in range(batch_size)]
            # generate until all inputs are completed
            for step in range(self.config.max_turns):
                if len(active_indices) == 0:
                    break

                # only process the active inputs
                question_i = prompts.non_tensor_batch['raw_prompt_ids'][active_indices]
                memory_i = [memory[idx] for idx in active_indices]
                messages = [
                    self.token_message_template.format(
                        question=question,
                        memory=memory if memory is not None else self.NO_MEMORY_TOKENS,
                    )
                    for question, memory in zip(question_i, memory_i)
                ]
                init_inputs = copy.deepcopy(messages)
                return_input_ids += copy.deepcopy(messages)
                result_mask_list = [[] for _ in range(len(messages))]
                # sample_index = torch.arange(batch_size, dtype=torch.long)[active_indices]
                # final_mask = torch.full(sample_index.shape, False, dtype=torch.bool)
                # input_ids = pad_tensor_list_to_length(messages, 
                #                                 pad_token_id=pad_token_id,
                #                                 max_length=self.max_input_length, 
                #                                 left_pad=True)
                # attention_masks = create_attention_mask(input_ids, pad_token_id=pad_token_id)
                # position_ids = create_position_ids(attention_masks)

                # active_inputs = [curr_inputs[i] for i in active_indices]
                # active_max_tokens = [curr_max_tokens[i] for i in active_indices]
                
                with self.update_sampling_params(
                    n=1, 
                    max_tokens=512,
                    stop_token_ids=[151644],
                    top_p=0.99,
                    top_k=10000
                ):  # 512 at most, and add <|im_start|> as stop for corner case
                    vllm_inputs = [{
                        'prompt_token_ids': raw_prompt_ids
                    } for raw_prompt_ids in messages]
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )

                # collect all tool calls
                tool_calls_list: List[List[str]] = []
                call_indices: List[int] = []

                # process each output
                # new_active_indices = []
                new_active_indices_in_active_indices = []
                for i, idx in enumerate(active_indices):
                    output_ids = outputs[i].outputs[0].token_ids
                    finish_reason = outputs[i].outputs[0].finish_reason
                    stop_reason = outputs[i].outputs[0].stop_reason
                    
                    if finish_reason == 'stop' and (stop_reason == None or stop_reason == self.tokenizer.pad_token_id):
                        # curr_inputs[idx] += output_ids
                        result_mask_list[i] += [1] * len(output_ids)
                        messages[i] += output_ids

                        output_str = self.tokenizer.decode(output_ids)
                        tool_calls: List[str] = self.extract_tool_calls(output_str)
                        if tool_calls:
                            tool_calls_list.append(tool_calls)
                            call_indices.append(i)
                            # new_active_indices.append(idx)
                            new_active_indices_in_active_indices.append(i)
                        else:
                            pass # no tool calls
                    elif finish_reason == 'length':
                        # output over max tokens
                        # curr_inputs[idx] += output_ids
                        messages[i] += output_ids
                        result_mask_list[i] += [1] * len(output_ids)
                    elif finish_reason == 'stop' and stop_reason == 151644: # 151644 is the id of <|im_start|>, is a illigal stop, we stop here
                        # curr_inputs[idx] += output_ids
                        messages[i] += output_ids
                        result_mask_list[i] += [1] * len(output_ids)
                    else:
                        raise ValueError(f"unknown stop reason. finish_reason: {finish_reason}, stop_reason: {stop_reason}")

                # batch process tool calls
                if tool_calls_list:
                    # Only tp_rank 0 executes the tools
                    if self.tp_rank == 0:
                        active_env_list = [env_list[i] for i in call_indices]
                        tool_responses_list = self.batch_execute(active_env_list, tool_calls_list)
                        
                        # Prepare data for broadcasting
                        broadcast_data = {
                            'tool_calls_list': tool_calls_list,
                            'call_indices': call_indices,
                            'tool_responses_list': tool_responses_list
                        }
                    else:
                        broadcast_data = None
                    
                    broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)
                    
                    # All ranks process the broadcasted data
                    if broadcast_data is not None:
                        tool_calls_list = broadcast_data['tool_calls_list']
                        call_indices = broadcast_data['call_indices']
                        tool_responses_list = broadcast_data['tool_responses_list']

                        for idx, tool_calls, tool_responses in zip(call_indices, tool_calls_list, tool_responses_list):
                            tool_response_str = ''
                            for call, response in zip(tool_calls, tool_responses):
                                tool_response_str += f"<tool_response>{call}\n{response}\n</tool_response>\nPlease read the tool response carefully and update the memory with new information that helps to answer the question, while retaining all relevant details from the previous memory. Return the updated memory within <memory> </memory> tags."
                            tool_response_str = "\n<|im_start|>user\n" + tool_response_str + "<|im_end|>"
                            output_ids = self.tokenizer.encode(tool_response_str)
                            # curr_inputs[idx] += output_ids
                            messages[idx] += output_ids
                            result_mask_list[idx] += [0] * len(output_ids)

                            # curr_inputs[idx] += self.gen_ids
                            messages[idx] += self.gen_ids
                            result_mask_list[idx] += [0] * len(self.gen_ids)

                length_checked_active_indices = []
                for i in new_active_indices_in_active_indices:
                    if len(messages[i])- len(init_inputs[i]) > self.config.response_length:
                        messages[i] = init_inputs[i] + messages[i][len(init_inputs[i]):len(init_inputs[i])+self.config.response_length]
                        result_mask_list[i] = result_mask_list[i][:self.config.response_length]
                    else:
                        length_checked_active_indices.append(i)
                new_active_indices_in_active_indices = copy.deepcopy(length_checked_active_indices)
                new_active_indices = [active_indices[i] for i in new_active_indices_in_active_indices]

                new_messages = [messages[i] for i in new_active_indices_in_active_indices]
                new_result_mask_list = [result_mask_list[i] for i in new_active_indices_in_active_indices]

                with self.update_sampling_params(
                    n=1, 
                    max_tokens=512,
                    stop_token_ids=[151644],
                    top_p=0.99,
                    top_k=10000
                ):  # 512 at most, and add <|im_start|> as stop for corner case
                    vllm_inputs = [{
                        'prompt_token_ids': raw_prompt_ids
                    } for raw_prompt_ids in new_messages]
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )

                new_new_active_indices = []
                for i, idx in enumerate(new_active_indices):
                    output_ids = outputs[i].outputs[0].token_ids
                    finish_reason = outputs[i].outputs[0].finish_reason
                    stop_reason = outputs[i].outputs[0].stop_reason
                    
                    if finish_reason == 'stop' and (stop_reason == None or stop_reason == self.tokenizer.pad_token_id):
                        # curr_inputs[idx] += output_ids
                        new_result_mask_list[i] += [1] * len(output_ids)
                        new_messages[i] += output_ids

                        output_str = self.tokenizer.decode(output_ids)
                        # tool_calls: List[str] = self.extract_tool_calls(output_str)
                        memory_str = self.extract_memory(output_str)
                        if memory_str and len(memory_str) > 0:
                            memory[idx] = self.tokenizer(memory_str)['input_ids']
                            new_new_active_indices.append(idx)
                        else:
                            pass
                    elif finish_reason == 'length':
                        # output over max tokens
                        # curr_inputs[idx] += output_ids
                        new_messages[i] += output_ids
                        new_result_mask_list[i] += [1] * len(output_ids)
                    elif finish_reason == 'stop' and stop_reason == 151644: # 151644 is the id of <|im_start|>, is a illigal stop, we stop here
                        # curr_inputs[idx] += output_ids
                        new_messages[i] += output_ids
                        new_result_mask_list[i] += [1] * len(output_ids)
                    else:
                        raise ValueError(f"unknown stop reason. finish_reason: {finish_reason}, stop_reason: {stop_reason}")

                for i, idx in enumerate(new_active_indices_in_active_indices):
                    messages[idx] = new_messages[i]
                    result_mask_list[idx] = new_result_mask_list[i]

                # check if need to truncate, if yes, truncate, and remove from active; if no, update curr_max_tokens
                length_checked_active_indices = []
                for i in range(len(active_indices)):
                    if len(result_mask_list[i]) > self.config.response_length:
                        messages[i] = init_inputs[i] + messages[i][len(init_inputs[i]):len(init_inputs[i])+self.config.response_length]
                        result_mask_list[i] = result_mask_list[i][:self.config.response_length]
                    elif len(memory[active_indices[i]]) > self.config.memory_length:
                        print(f"[Warning] Memory length exceeded (length={len(memory[active_indices[i]])}, limit={self.config.memory_length})")
                    else:
                        if active_indices[i] in new_new_active_indices:
                            length_checked_active_indices.append(active_indices[i])

                return_seq += messages
                return_result_mask_list += result_mask_list
                if step == self.config.max_turns -1:
                    final_mask += [True] * len(active_indices)
                else:
                    final_mask += [active_indices[i] not in length_checked_active_indices for i in range(len(active_indices))]
                sample_index += prompts.batch['sample_index'][active_indices]

                active_indices = length_checked_active_indices

                # for idx in active_indices:
                #     # assert len(curr_inputs[idx]) - len(init_inputs[idx]) == len(result_mask_list[idx]), f"curr_inputs: {len(curr_inputs[idx])}, init_inputs: {len(init_inputs[idx])}, result_mask_list: {len(result_mask_list[idx])}"
                #     if len(curr_inputs[idx]) - len(init_inputs[idx]) >= self.config.response_length:
                #         curr_inputs[idx] = init_inputs[idx] \
                #             + curr_inputs[idx][len(init_inputs[idx]):len(init_inputs[idx])+self.config.response_length]
                #         result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                #     else:
                #         curr_max_tokens[idx] = self.config.response_length - len(curr_inputs[idx]) + len(init_inputs[idx])
                #         if idx in new_active_indices:
                #             length_checked_active_indices.append(idx)
                # active_indices = length_checked_active_indices

            return_output_ids = []
            for input_ids, seq, result_mask in zip(return_input_ids, return_seq, return_result_mask_list):
                return_output_ids.append(seq[len(input_ids):])
                assert len(result_mask) == len(seq) - len(input_ids), f"length mismatch: len(result_mask)={len(result_mask)}，len(seq)={len(seq)}，len(input_ids)={len(input_ids)}"

            # collect the all rollouts
            # for i, input_ids in enumerate(idx_list):
            #     for j in range(self.sampling_params.n):
            #         idx = i * self.sampling_params.n + j
            #         input_len = len(input_ids)
            #         output_ids_list.append(curr_inputs[idx][input_len:])

        response_attention_mask_list = []
        response_list = []
        result_mask_list_padded = []
        attention_mask_list = []
        ori_input_ids_list = []
        # position_ids = []
        device = prompts.batch['input_ids'].device
        for output_ids, result_mask, input_ids in zip(return_output_ids, return_result_mask_list, return_input_ids):
            assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
            # to tensor 
            response = torch.tensor(output_ids, device=device)
            result_mask = torch.tensor(result_mask, device=device)
            ori_input_ids = torch.tensor(input_ids, device=device)
            # response attention mask, 1 for valid, 0 for invalid
            response_attention_mask = torch.ones_like(response, dtype=torch.int64)
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
            response_attention_mask_list.append(response_attention_mask)
            attention_mask = torch.ones_like(ori_input_ids, dtype=torch.int64)
            attention_mask = pad_sequence_to_length(attention_mask, self.max_input_length, 0, True)
            attention_mask_list.append(attention_mask)
            # response, pad to response_length
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            response_list.append(response)
            ori_input_ids = pad_sequence_to_length(ori_input_ids, self.max_input_length, self.pad_token_id, True)
            ori_input_ids_list.append(ori_input_ids)
            # result mask, 1 for non-result, 0 for result or pad
            result_mask = pad_sequence_to_length(result_mask, self.config.response_length, 0)
            result_mask_list_padded.append(result_mask)
        response_attention_mask = torch.stack(response_attention_mask_list, dim=0)
        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)
        ori_input_ids = torch.stack(ori_input_ids_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)

        # if self.config.n > 1 and do_sample:
        #     ori_input_ids = ori_input_ids.repeat_interleave(self.config.n, dim=0)
        #     attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
        #     position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            # batch_size = batch_size * self.config.n
        batch_size = len(final_mask)
        seq = torch.cat([ori_input_ids, response], dim=-1)

        # response_length = response.size(1)
        # delta_position_id = torch.arange(1, response_length + 1, device=prompts.batch['position_ids'].device)
        # delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        # response_position_ids = position_ids[:, -1:] + delta_position_id
        # position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
                
        # concat attenion_mask for input and response
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        position_ids = get_position_ids(attention_mask).to(prompts.batch['position_ids'].device)

        # result mask: result part is 0, other part is 1
        loss_mask = result_mask * response_attention_mask
        final_mask = torch.tensor(final_mask)
        sample_index = torch.tensor(sample_index)
        
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict({
            'prompts': ori_input_ids,
            'responses': response,
            'input_ids': seq,  # here input_ids become the whole sentences
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'final_mask': final_mask,
            'sample_index': sample_index
        }, batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)