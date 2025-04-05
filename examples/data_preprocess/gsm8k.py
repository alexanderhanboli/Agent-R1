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
Preprocess the GSM8k dataset to parquet format
"""

import json
import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from agent_r1.tool.tools import _default_tools


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/gsm8k')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'openai/gsm8k'

    dataset = datasets.load_dataset(data_source, 'main')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = """For any assistant response, please always first conduct reasoning within <think></think> XML tags.

If it is necessary to use any tool, you should put the function name and arguments within <tool_call></tool_call> XML tags. You can only use the tools provided to you. For example:
<think> [Your reasoning here] </think>
<tool_call> [Function name and arguments] </tool_call>

Otherwise, if you are ready to output your answer, please put it inside <answer></answer> XML tags. In your answer, put the final numerical solution in \\boxed{...}. For example:
<think> [Your reasoning here] </think>
<answer> Alex has 10 apples, and he gives 2 to his friend, so he has \\boxed{8} apples left. </answer>

Now, follow the above instructions to answer the following question:
"""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')

            question = instruction_following + question_raw

            answer_raw = example.pop('answer')
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                {
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # train_dataset = train_dataset.select(range(64))
    test_dataset = test_dataset.select(range(100))

    # pretty print a json of one item
    print(json.dumps(train_dataset[0], indent=4))

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    # Example of how the processed data is used with ToolRLDataset
    print("\nDemonstrating how the processed data is used with ToolRLDataset:")
    print("-" * 80)
    
    from agent_r1.src.agent_rl_dataset import ToolRLDataset
    from transformers import AutoTokenizer
    from agent_r1.tool.tool_env import ToolEnv
    
    # Initialize tokenizer and tool environment
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tool_env = ToolEnv(tools=_default_tools(['python']))
    
    # Create dataset using the processed parquet file
    dataset = ToolRLDataset(
        parquet_files=os.path.join(local_dir, 'train.parquet'),
        tokenizer=tokenizer,
        prompt_key="prompt",
        max_prompt_length=2048,
        tool_env=tool_env
    )
    
    # Get first item and show the tokenized output
    print("\nExample of first item after processing through ToolRLDataset:")
    item = dataset[0]
    decoded = tokenizer.decode(item['input_ids'])
    print("\nDecoded tokenized input:")
    print("-" * 80)
    print(decoded)
    print("-" * 80)
    
    # Print attention mask shape and other relevant info
    print("\nInput shapes:")
    print(f"input_ids shape: {item['input_ids'].shape}")
    print(f"attention_mask shape: {item['attention_mask'].shape}")
    print(f"position_ids shape: {item['position_ids'].shape}")
