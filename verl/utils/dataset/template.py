re_call_template_sys = """In this environment you have access to a set of tools you can use to assist with the user query. \
You may perform multiple rounds of function calls. \
In each round, you can call one or more functions.

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

re_call_template_sys = """<|im_start|>system
In this environment you have access to a set of tools you can use to assist with the user query. \
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
</tool_call><|im_end|>
<|im_start|>user
<question> {question} </question>

<memory> {memory} </memory><|im_end|>
<|im_start|>assistant
<think>"""

prompt_template_dict = {}
prompt_template_dict['re_call_template_sys'] = re_call_template_sys
