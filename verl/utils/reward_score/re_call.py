import re
import string
from typing import Union, List
from collections import Counter

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])
    
    return final_metric['f1']

def validate_template_format(text: str) -> tuple[bool, str]:
    """
    validate the template format
    return: (is valid, error message)
    """
    # extract all assistant responses
    assistant_responses = []
    current_pos = 0
    while True:
        start_pos = text.find("<|im_start|>assistant\n", current_pos)
        if start_pos == -1:
            break
        end_pos = text.find("<|im_end|>", start_pos)
        if end_pos == -1:
            break
        response = text[start_pos + len("<|im_start|>assistant\n"):end_pos].strip()
        assistant_responses.append(response)
        current_pos = end_pos + len("<|im_end|>")

    if not assistant_responses:
        return False, "no assistant response found"

    for i, response in enumerate(assistant_responses):
        # 1. check <think> and </think> pair
        think_count = response.count("<think>")
        think_end_count = response.count("</think>")
        if think_count != think_end_count:
            return False, f"<think> and </think> are not paired: think={think_count}, think_end={think_end_count}"
        if i == 0 and think_count == 0:
            return False, "missing <think> tag"

        # 2. check <tool_call> and </tool_call> pair
        tool_call_count = response.count("<tool_call>")
        tool_call_end_count = response.count("</tool_call>")
        if tool_call_count != tool_call_end_count:
            return False, f"<tool_call> and </tool_call> are not paired: tool_call={tool_call_count}, tool_call_end={tool_call_end_count}"

        # 3. check the content of each tool_call can be parsed as json
        if i == 0:
            current_pos = 0
            while True:
                tool_call_start = response.find("<tool_call>", current_pos)
                if tool_call_start == -1:
                    break
                tool_call_end = response.find("</tool_call>", tool_call_start)
                if tool_call_end == -1:
                    break
                
                tool_call_content = response[tool_call_start + len("<tool_call>"):tool_call_end].strip()
                
                # check if it contains name and arguments
                if '"name"' not in tool_call_content or '"arguments"' not in tool_call_content:
                    return False, "tool_call is missing name or arguments field"
                
                try:
                    import json
                    json.loads(tool_call_content)
                except json.JSONDecodeError:
                    return False, f"tool_call is not a valid json: {tool_call_content}"
                
                current_pos = tool_call_end + len("</tool_call>")

    # 4. check if the last response contains \\boxed
    if "\\box" not in assistant_responses[-1]:
        return False, "the last response is missing \\boxed"

    return True, assistant_responses[-1]

def compute_score_with_format(tokenizer, solution_str, ground_truth) -> tuple[float, str]:
    if not solution_str.endswith(tokenizer.eos_token):
        return 0, f'not end with eos token'
    
    valid_template, reason = validate_template_format(solution_str)
    if not valid_template:
        return 0, f'bad format: {reason}'
    else:
        response = reason

    try:
        answer = remove_boxed(last_boxed_only_string(response))
    except Exception as e:
        return 0, f'find box error: {e}'

    f1_score = get_f1_score(answer, ground_truth)
    if f1_score > 0:
        return f1_score, f'correct answer, get f1 score: {f1_score}'
    else:
        return 0.1, f'wrong answer but good format: {answer}'