import unittest
import re
from agent_r1.src.reward_score.gsm8k import (
    answer_check,
    extract_solution,
    compute_score_format,
    compute_score_answer,
    compute_score_format_answer
)

class TestGSM8K(unittest.TestCase):
    def test_answer_check(self):
        solution_str = "<|im_start|>user\nWhat is 123.5?\n<|im_end|>\n<|im_start|>assistant\n<answer> After calculating, we get \\boxed{123.5} </answer>\n<|im_end|>"
        self.assertTrue(answer_check(extract_solution(solution_str), "123.5"))

        solution_str = "<|im_start|>user\nWhat is 123.5?\n<|im_end|>\n<|im_start|>assistant\nAfter calculating, we get \\boxed{123.5}\n<|im_end|>"
        self.assertFalse(answer_check(extract_solution(solution_str), "123.5"))
        
    def test_extract_solution(self):
        # Test valid boxed content
        self.assertEqual(extract_solution("<|im_start|>user\nWhat is 42?\n<|im_end|>\n<|im_start|>assistant\n<answer> The answer is \\boxed{42} </answer>\n<|im_end|>"), "42")
        self.assertEqual(extract_solution("<|im_start|>user\nWhat is 123.5?\n<|im_end|>\n<|im_start|>assistant\n<answer> \\boxed{123.5} </answer>\n<|im_end|>"), "123.5")
        self.assertEqual(extract_solution("<|im_start|>user\nWhat is -15?\n<|im_end|>\n<|im_start|>assistant\n<answer> Complex calculation: \\boxed{-15} </answer>\n<|im_end|>"), "-15")
        
        # Test invalid or missing boxed content
        self.assertIsNone(extract_solution("No boxed content here"))
        self.assertIsNone(extract_solution(""))
        self.assertIsNone(extract_solution("\\boxed{"))
        self.assertIsNone(extract_solution("boxed{42}"))

    def test_compute_score_format(self):
        # Test perfect format
        perfect_solution = """<|im_start|>user
What is 42?\n<|im_end|>
<|im_start|>assistant
<think>Let's solve this step by step.</think>
<tool_call>calculator</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>42</tool_response>
<|im_end|>
<|im_start|>assistant
<think>Now I can provide the answer.</think>
<answer>The answer is \\boxed{42}</answer>
<|im_end|>"""
        self.assertEqual(compute_score_format(perfect_solution), 1.0)

        # Test partial format (missing think tags)
        partial_solution = """<|im_start|>user
What is 42?\n<|im_end|>
<|im_start|>assistant
Let's solve this.
<tool_call>calculator</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>42</tool_response>
<|im_end|>
<|im_start|>assistant
The answer is \\boxed{42}
<|im_end|>"""
        self.assertEqual(compute_score_format(partial_solution), 0.0)

        # Test invalid format
        self.assertEqual(compute_score_format(None), 0.0)
        self.assertEqual(compute_score_format(""), 0.0)

    # Test partial format reward with final assistant block correct
    def test_compute_score_format_final_assistant_block_correct(self):
        solution_str = """<|im_start|>user
What is 42?\n<|im_end|>
<|im_start|>assistant
<tool_call>calculator</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>42</tool_response>
<|im_end|>
<|im_start|>assistant
<think>Now I can provide the answer.</think>
<answer>The answer is \\boxed{42}</answer>
<|im_end|>"""
        self.assertEqual(compute_score_format(solution_str), 0.5)

    def test_compute_score_answer(self):
        # Test correct answer
        correct_solution = """<|im_start|>user
What is 42?\n<|im_end|>
<|im_start|>assistant
<think>Here's the solution.</think>
<answer>The answer is \\boxed{42}</answer>
<|im_end|>"""
        self.assertEqual(compute_score_answer(correct_solution, "42"), 1.0)

        # Test incorrect answer
        incorrect_solution = """<|im_start|>user
What is 42?\n<|im_end|>
<|im_start|>assistant
<think>Here's the solution.</think>
<answer>The answer is \\boxed{43}</answer>
<|im_end|>"""
        self.assertEqual(compute_score_answer(incorrect_solution, "42"), 0.0)

        # Test invalid input
        self.assertEqual(compute_score_answer(None, "42"), 0.0)
        self.assertEqual(compute_score_answer("", "42"), 0.0)

    def test_compute_score_format_answer(self):
        # Test perfect format and correct answer
        perfect_solution = """<|im_start|>user
What is 42?\n<|im_end|>
<|im_start|>assistant
<think>Let's solve this step by step.</think>
<tool_call>calculator</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>42</tool_response>
<|im_end|>
<|im_start|>assistant
<think>Now I can provide the answer.</think>
<answer>The answer is \\boxed{42}</answer>
<|im_end|>"""
        self.assertEqual(compute_score_format_answer(perfect_solution, "42"), 1.0)

        # Test perfect format but wrong answer
        wrong_answer_solution = """<|im_start|>user
What is 42?\n<|im_end|>
<|im_start|>assistant
<think>Let's solve this step by step.</think>
<tool_call>calculator</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>43</tool_response>
<|im_end|>
<|im_start|>assistant
<think>Now I can provide the answer.</think>
<answer>The answer is \\boxed{43}</answer>
<|im_end|>"""
        self.assertEqual(compute_score_format_answer(wrong_answer_solution, "42"), 0.2)

        # Test neither format nor answer is correct
        wrong_format_answer_solution = """<|im_start|>user
What is 42?\n<|im_end|>
<|im_start|>assistant
<tool_call>calculator</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>43</tool_response>
<|im_end|>
<|im_start|>assistant
<think>Now I can provide the answer.</think>
<answer>The answer is \\boxed{43}</answer>
<|im_end|>"""
        self.assertEqual(compute_score_format_answer(wrong_format_answer_solution, "42"), 0.1)

        # Test correct answer but wrong format
        correct_answer_wrong_format_solution = """<|im_start|>user
What is 42?\n<|im_end|>
<|im_start|>assistant
<think>Here's the solution.</think>
<answer>The answer is \\boxed{42}</answer>
<|im_end|>"""
        self.assertEqual(compute_score_format_answer(correct_answer_wrong_format_solution, "42"), 0.9)
        # Test invalid input
        self.assertEqual(compute_score_format_answer(None, "42"), 0.0)
        self.assertEqual(compute_score_format_answer("", "42"), 0.0)
        self.assertEqual(compute_score_format_answer(perfect_solution, None), 0.0)

if __name__ == '__main__':
    unittest.main() 