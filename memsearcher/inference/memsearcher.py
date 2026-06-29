import os
import re
import json
import requests
import time
from typing import List
from functools import wraps

def retry(max: int=10, sleep: int=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[retry] try {i} times")
                    if i == max - 1:
                        raise Exception("Retry {} failed after {} times".format(func.__name__, max))
                    elif sleep:
                        time.sleep(sleep)
        return wrapper
    return decorator

class MemSearcher():
    system_prompt = """In this environment you have access to a set of tools you can use to assist with the user query. \
You may perform multiple rounds of function calls. \
In each round, you can call one or more functions. \

Here are available functions in JSONSchema format: \n```json\n{func_schemas}\n```

In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. \
The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags. \
The results of the function calls will be given back to you after execution, \
and you can continue to call functions until you get the final answer for the user's question. \
Finally, if you have got the answer, enclose it within \\boxed{{}} with latex format and do not continue to call functions, \
i.e., <think> Based on the response from the function call, I get the weather information. </think> The weather in Beijing on 2025-04-01 is \\[ \\boxed{{20C}} \\].

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

    system_prompt = """In this environment you have access to a set of tools you can use to assist with the user query. \
You will answer a complex question through iterative reasoning, tool call, and memory update. 

Here are available functions in JSONSchema format: 
```json
[
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia for a given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search for."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of results to return. The default value is 5.",
                        "default": 5
                    }
                },
                "required": [
                    "query"
                ]
            }
        }
    }
]
```

You will be presented with the question and a previous memory. \
You need to first think about the reasoning process within <think> </think> tags and then choose one:
1. If you have got the answer from the previous memory, enclose it within \\boxed{} with latex format and do not call functions, \
i.e., <think> Based on the previous memory, I get the weather information. </think> The weather in Beijing on \
2025-04-01 is \\[ \\boxed{20C} \\].
2. If you find you lack some knowledge to solve the question, conduct function calling to get the information. \
The function calling is enclosed within <tool_call> </tool_call> tags. The results of the function calls will be given back to you after execution. \
Please read the results carefully and update the memory with new information that helps to answer the question, while retaining all \
relevant details from the previous memory.

For function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

    user_prompt = """<question> {question} </question>

<memory> {memory} </memory>"""

    def __init__(self, model_url, executor_url, max_turns):
        self.model_url = model_url
        self.executor_url = executor_url
        self.max_turns = max_turns
        
    def init_prompt(self, func_schemas, question, memory=""):
        # system_prompt = f"<|im_start|>system\n{self.system_prompt.format(func_schemas=func_schemas)}<|im_end|>"
        system_prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>"
        # user_prompt = f"<|im_start|>user\n{question}<|im_end|>"
        user_prompt = "<|im_start|>user\n{}<|im_end|>".format(self.user_prompt.format(question=question, memory=memory))
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return system_prompt + "\n" + user_prompt + "\n" + assistant_prefix

    def cat_assistant_response(self, curr_prompt, assistant_response):
        return curr_prompt + assistant_response + "<|im_end|>"
    
    def cat_tool_results(self, curr_prompt, tool_calls, results):
        tool_response_str = ""
        for tool_call, result in zip(tool_calls, results):
            tool_response_str += f"<tool_response>{tool_call}\n{result}\n</tool_response>\nPlease read the tool response carefully and update the memory with new information that helps to answer the question, while retaining all relevant details from the previous memory. Return the updated memory within <memory> </memory> tags."
        tool_response_str = f"<|im_start|>user\n{tool_response_str}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n"
        return curr_prompt + "\n" + tool_response_str + "\n" + assistant_prefix

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
    
    def execute_tool_calls(self, env: str, tool_calls: List[str]) -> List[str]:
        def exe_tool_call(env, call):
            url = self.executor_url + '/execute'

            call_str = self.format_tool_call(call)
            if call_str.startswith("error: parse tool call failed"):
                return call_str

            try:
                data = {
                    'env': env,
                    'call': call_str
                }
                response = requests.post(url, json=data, timeout=3)
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

        results = []
        for tool_call in tool_calls:
            # result = exe_tool_call(env, tool_call)
            result = wikipedia_search(tool_call)
            results.append(result)
        return results
    
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
        
    @retry(max=200, sleep=1)
    def run(self, env, func_schemas, question, tokenizer=None):
        temp = float(os.environ.get("MEMSEARCHER_TEMPERATURE", "0.0"))
        lengths = []
        curr_prompt = self.init_prompt(func_schemas, question)
        for i in range(self.max_turns):
            response = requests.post(
                f'{self.model_url}/generate', 
                json={
                    "text": curr_prompt,
                    "sampling_params": {
                        "temperature": temp,
                        "max_new_tokens": 512
                    }
                }
            ).json()
            # try:
            curr_prompt = self.cat_assistant_response(curr_prompt, response['text'])
            # except Exception as e:
            #     print(f"Unexpected error: {e}, response: {response}")
            #     return curr_prompt

            tool_calls: List[str] = self.extract_tool_calls(response['text'])
            lengths.append(len(tokenizer(curr_prompt)['input_ids']) if tokenizer is not None else 0)
            if len(tool_calls) == 0:
                break

            results: List[str] = self.execute_tool_calls(env, tool_calls)
            curr_prompt = self.cat_tool_results(curr_prompt, tool_calls, results)

            response = requests.post(
                f'{self.model_url}/generate', 
                json={
                    "text": curr_prompt,
                    "sampling_params": {
                        "temperature": temp,
                        "max_new_tokens": 512
                    }
                }
            ).json()
            # try:
            curr_prompt = self.cat_assistant_response(curr_prompt, response['text'])
            # print(curr_prompt)
            # except Exception as e:
            #     print(f"Unexpected error: {e}, response: {response}")
            #     return curr_prompt

            if tokenizer is not None:
                lengths[-1] = len(tokenizer(curr_prompt)['input_ids'])

            memory = self.extract_memory(response['text'])
            curr_prompt = self.init_prompt(func_schemas, question, memory)

            # lengths.append(len(tokenizer(curr_prompt)['input_ids']))
            # print(curr_prompt)

        # if i == 5:
        #     print(curr_prompt)
        # print(curr_prompt)
        return curr_prompt, lengths

class ReCall():
    system_prompt = """In this environment you have access to a set of tools you can use to assist with the user query. \
You may perform multiple rounds of function calls. \
In each round, you can call one or more functions. \

Here are available functions in JSONSchema format: \n```json\n{func_schemas}\n```

In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. \
The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags. \
The results of the function calls will be given back to you after execution, \
and you can continue to call functions until you get the final answer for the user's question. \
Finally, if you have got the answer, enclose it within \\boxed{{}} with latex format and do not continue to call functions, \
i.e., <think> Based on the response from the function call, I get the weather information. </think> The weather in Beijing on 2025-04-01 is \\[ \\boxed{{20C}} \\].

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

    def __init__(self, model_url, executor_url, max_turns):
        self.model_url = model_url
        self.executor_url = executor_url
        self.max_turns = max_turns

    def init_prompt(self, func_schemas, question):
        system_prompt = f"<|im_start|>system\n{self.system_prompt.format(func_schemas=func_schemas)}<|im_end|>"
        user_prompt = f"<|im_start|>user\n{question}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return system_prompt + "\n" + user_prompt + "\n" + assistant_prefix

    def cat_assistant_response(self, curr_prompt, assistant_response):
        return curr_prompt + assistant_response + "<|im_end|>"

    def cat_tool_results(self, curr_prompt, tool_calls, results):
        tool_response_str = ""
        for tool_call, result in zip(tool_calls, results):
            tool_response_str += f"<tool_response>{tool_call}\n{result}\n</tool_response>\n"
        tool_response_str = f"<|im_start|>user\n{tool_response_str}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return curr_prompt + "\n" + tool_response_str + "\n" + assistant_prefix

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

    def execute_tool_calls(self, env: str, tool_calls: List[str]) -> List[str]:
        def exe_tool_call(env, call):
            url = self.executor_url + '/execute'

            call_str = self.format_tool_call(call)
            if call_str.startswith("error: parse tool call failed"):
                return call_str

            try:
                data = {
                    'env': env,
                    'call': call_str
                }
                response = requests.post(url, json=data, timeout=3)
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

        results = []
        for tool_call in tool_calls:
            result = wikipedia_search(tool_call)
            results.append(result)
        return results

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

    @retry(max=5, sleep=1)
    def run(self, env, func_schemas, question, tokenizer=None):
        lengths = []
        curr_prompt = self.init_prompt(func_schemas, question)
        for _ in range(self.max_turns):
            response = requests.post(
                f'{self.model_url}/generate',
                json={
                    "text": curr_prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 512
                    }
                }
            ).json()
            try:
                curr_prompt = self.cat_assistant_response(curr_prompt, response['text'])
            except Exception as e:
                print(f"Unexpected error: {e}, response: {response}")
                return curr_prompt, lengths

            tool_calls: List[str] = self.extract_tool_calls(response['text'])
            lengths.append(len(tokenizer(curr_prompt)['input_ids']) if tokenizer is not None else 0)
            if len(tool_calls) == 0:
                break

            results: List[str] = self.execute_tool_calls(env, tool_calls)
            curr_prompt = self.cat_tool_results(curr_prompt, tool_calls, results)

            if tokenizer is not None:
                lengths[-1] = len(tokenizer(curr_prompt)['input_ids'])

        return curr_prompt, lengths
