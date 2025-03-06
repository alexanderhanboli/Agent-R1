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

import re
import json

def extract_solution(solution_str):
    if solution_str is None:
        return None
        
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        # 首先尝试从<answer>标签内提取数字
        answer_content = match.group(1)
        numbers_in_answer = re.findall(r"(\-?[0-9\\.\\,]+)", answer_content)
        
        if numbers_in_answer:
            # 从<answer>标签内找到数字
            invalid_str = ['', '.']
            # 找到最后一个非无效的数字
            for final_answer in reversed(numbers_in_answer):
                if final_answer not in invalid_str:
                    return final_answer
        
        # 如果<answer>标签内没有找到有效数字，则在整个solution_str中查找
        numbers_in_solution = re.findall(r"(\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if numbers_in_solution:
            invalid_str = ['', '.']
            # 找到最后一个非无效的数字
            for final_answer in reversed(numbers_in_solution):
                if final_answer not in invalid_str:
                    break
        return final_answer
    else:
        return None

def compute_score_format(solution_str):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    if solution_str is None:
        return 0.0
        
    try:
        # Perfect format match for the new structure
        # First <|im_start|>assistant should have <think> and possibly <tool_call>
        # Then <|im_start|>tool with <tool_response> (can repeat with assistant/tool pairs)
        # Final <|im_start|>assistant with the answer and <|im_end|>
        
        # Check for basic structure with <|im_start|>assistant and <|im_end|> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n?(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        tool_blocks = re.findall(r'<\|im_start\|>user\n?(.*?)<\|im_end\|>', solution_str, re.DOTALL)

        format_reward = 0.0
        
        # If no blocks found, return 0
        if not assistant_blocks:
            return 0.0
        
        # Perfect format requires at least one assistant block and matching tool blocks if tool calls exist
        # Check first assistant block contains <think> tags
        if len(assistant_blocks) > 1:
            if len(tool_blocks) == len(assistant_blocks):
                for i, assistant_block in enumerate(assistant_blocks[:-1]):
                    if assistant_block.count('<think>') == 1 and assistant_block.count('</think>') == 1 and assistant_block.count('<tool_call>') == 1 and assistant_block.count('</tool_call>') == 1:
                        format_match = re.search(r'^<think>(.*?)</think>\n<tool_call>(.*?)</tool_call>$', assistant_block, re.DOTALL)
                        if format_match:
                            tool_block = tool_blocks[i+1]
                            tool_response = re.search(r'<tool_response>(.*?)</tool_response>', tool_block, re.DOTALL)
                            if tool_response:
                                try:
                                    tool_result = json.loads(tool_response.group(1)).get('result', None)
                                except json.JSONDecodeError:
                                    tool_result = None
                                if tool_result:
                                    format_reward += 0.4
                            else:
                                format_reward += 0.2
                        else:
                            tool_block = tool_blocks[i+1]
                            tool_response = re.search(r'<tool_response>(.*?)</tool_response>', tool_block, re.DOTALL)
                            if tool_response:
                                try:
                                    tool_result = json.loads(tool_response.group(1)).get('result', None)
                                except json.JSONDecodeError:
                                    tool_result = None
                                if tool_result:
                                    format_reward += 0.2          
            else:
                print(f"Error in compute_score_format: {len(tool_blocks)} != {len(assistant_blocks) - 1}")
                format_reward = 0.0
        
        # Check the last assistant block contains <answer> tags
        if assistant_blocks:  # 确保有至少一个assistant块
            last_assistant_block = assistant_blocks[-1]
            think_answer_match = re.search(r'^<think>(.*?)</think>\n<answer>(.*?)</answer>$', last_assistant_block, re.DOTALL)
            think_match = re.search(r'<think>(.*?)</think>', last_assistant_block, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', last_assistant_block, re.DOTALL)
            if think_answer_match:
                format_reward += 0.2
            # elif think_match and answer_match:
            #     format_reward += 0.15
            # elif think_match and not answer_match:
            #     format_reward += 0.1
            # elif not think_match and answer_match:
            #     format_reward += 0.05
        
        return format_reward
    except Exception as e:
        print(f"Error in compute_score_format: {e}")
        return 0.0

def compute_score_answer(solution_str, ground_truth):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    if solution_str is None:
        return 0.0
        
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n?(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        tool_blocks = re.findall(r'<\|im_start\|>user\n?(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        
        if not assistant_blocks or not tool_blocks:
            return 0.0
        else:
            # 使用最后一个assistant块
            last_block = assistant_blocks[-1]
            answer = extract_solution(solution_str=last_block)
            tool_block = tool_blocks[-1]
            tool_response = re.search(r'<tool_response>\n(.*?)\n</tool_response>', tool_block, re.DOTALL)
            if tool_response:
                try:
                    tool_result = json.loads(tool_response.group(1))
                except json.JSONDecodeError:
                    tool_result = None
                if tool_result:
                    tool_result = tool_result.get('result', None)
            else:
                return 0.0

        if answer is None:
            return 0.0
        else:
            if answer == ground_truth and answer == tool_result:
                return 1.0
            else:
                return 0.0
    except Exception as e:
        print(f"Error in compute_score_answer: {e}")
        return 0.0
        
def compute_score_format_answer(solution_str, ground_truth):
    """The scoring function for GSM8k.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    
    print(f"[DEBUG] solution_str: {solution_str}")
    print(f"[DEBUG] ground_truth: {ground_truth}")
    try:
        format_score = compute_score_format(solution_str)
        answer_score = compute_score_answer(solution_str, ground_truth)
        print(f"[DEBUG] format_score: {format_score}, answer_score: {answer_score}")
        # return format_score + answer_score
        return format_score
    except Exception as e:
        print(f"Error in compute_score_format_answer: {e}")
        return 0.0

def compute_score_em(solution_str, ground_truth):
    """The scoring function for GSM8k.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    if solution_str is None or ground_truth is None:
        return 0.0
        
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n?(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        
        if not assistant_blocks:
            return 0.0
        else:
            # 使用最后一个assistant块
            last_block = assistant_blocks[-1]
            answer = extract_solution(solution_str=last_block)
            
        if answer == ground_truth:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        print(f"Error in compute_score_em: {e}")
        return 0.0