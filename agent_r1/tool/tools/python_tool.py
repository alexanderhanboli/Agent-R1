"""
Python code interpreter tool implementation
"""

import json
import logging
import sys
import traceback
import warnings
from typing import Dict
from io import StringIO
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from agent_r1.tool.tool_base import Tool

class PythonTool(Tool):
    """
    Tool for executing Python code while maintaining state between executions
    """
    
    def __init__(self):
        """
        Initialize the Python code interpreter tool
        """
        name = "python"
        description = "Execute Python code with support for data analysis, visualization, and numerical computations. Variables defined in previous executions are preserved for use in subsequent code."
        parameters = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. The code can use pandas (pd), numpy (np), and matplotlib.pyplot (plt). Variables defined in this code will be available in subsequent executions."
                }
            },
            "required": ["code"]
        }
        
        super().__init__(name, description, parameters)
        self.logger = logging.getLogger(__name__)
        
        # Initialize the execution environment with common libraries
        self.globals = {
            "numpy": np,
            "np": np,
            "pd": pd,
            "plt": plt,
            "math": __import__('math'),
        }

    def execute(self, args: Dict) -> str:
        """
        Execute Python code while maintaining state
        
        Args:
            args: Tool parameters containing:
                - "code": Python code to execute
            
        Returns:
            JSON string containing execution results:
            - stdout: Standard output
            - stderr: Standard error
            - error: Error message if execution failed
        """
        code = args.get("code", "").strip()
        if not code:
            return json.dumps({"error": "No code provided"})
        
        # Redirect stdout and stderr
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        try:
            # Execute the code in the global namespace
            # This ensures variables persist between executions
            exec(code, self.globals, self.globals)
            
            result = {
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue()
            }
            return json.dumps(result)
            
        except Exception as e:
            error_msg = traceback.format_exc()
            self.logger.error(f"Code execution failed: {error_msg}")
            return json.dumps({
                "error": str(e),
                "traceback": error_msg,
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue()
            })
            
        finally:
            # Restore stdout and stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for code execution
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value based on execution success
        """
        try:
            result_dict = json.loads(result)
            
            # If there was an error, give minimal reward
            if "error" in result_dict:
                return 0.05
            
            # Base reward for successful execution
            reward = 0.3
            
            # Additional reward for code complexity
            code = args.get("code", "")
            try:
                import ast
                ast_tree = ast.parse(code)
                num_nodes = sum(1 for _ in ast.walk(ast_tree))
                reward += min(0.1, 0.001 * num_nodes)
            except:
                pass
            
            return min(0.4, reward)  # Cap at 0.4
            
        except:
            return 0.0  # Invalid result format

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create tool instance
    python_tool = PythonTool()
    
    def print_result(result_str: str):
        """Helper function to print execution results"""
        result = json.loads(result_str)
        print("\nExecution Result:")
        print("-" * 50)
        if "error" in result:
            print("Error:", result["error"])
            if "traceback" in result:
                print("\nTraceback:")
                print(result["traceback"])
        if result.get("stdout"):
            print("Standard Output:")
            print(result["stdout"].rstrip())
        if result.get("stderr"):
            print("\nStandard Error:")
            print(result["stderr"].rstrip())
        print("-" * 50)
    
    # Test Case 1: Basic variable definition and persistence
    print("\nTest Case 1: Basic variable definition and persistence")
    test1_code1 = """
x = 42
y = "Hello, World!"
print(f"Defined x = {x} and y = {y}")
"""
    print_result(python_tool.execute({"code": test1_code1}))
    
    test1_code2 = """
print(f"Previous values: x = {x}, y = {y}")
x += 8
print(f"Updated x = {x}")
"""
    print_result(python_tool.execute({"code": test1_code2}))
    
    # Test Case 2: NumPy and mathematical operations
    print("\nTest Case 2: NumPy and mathematical operations")
    test2_code = """
arr = np.array([1, 2, 3, 4, 5])
mean = np.mean(arr)
std = np.std(arr)
print(f"Array: {arr}")
print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std:.2f}")
"""
    print_result(python_tool.execute({"code": test2_code}))
    
    # Test Case 3: Pandas DataFrame operations
    print("\nTest Case 3: Pandas DataFrame operations")
    test3_code1 = """
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'London', 'Paris']
}
df = pd.DataFrame(data)
print("Created DataFrame:")
print(df)
"""
    print_result(python_tool.execute({"code": test3_code1}))
    
    test3_code2 = """
df['country'] = ['USA', 'UK', 'France']
print("Updated DataFrame with new column:")
print(df)
"""
    print_result(python_tool.execute({"code": test3_code2}))
    
    # Test Case 4: Error handling
    print("\nTest Case 4: Error handling")
    test4_code = """
# This will raise a NameError
print(undefined_variable)
"""
    print_result(python_tool.execute({"code": test4_code}))
    
    # Test Case 5: Complex computation with multiple libraries
    print("\nTest Case 5: Complex computation with multiple libraries")
    test5_code = """
# Create some sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Calculate some statistics
stats = {
    'mean': np.mean(y),
    'max': np.max(y),
    'min': np.min(y)
}

print("Data Statistics:")
for key, value in stats.items():
    print(f"{key}: {value:.3f}")

# The variables x, y, df, and stats will persist for future use
"""
    print_result(python_tool.execute({"code": test5_code}))
    
    # Test Case 6: Using previously defined variables
    print("\nTest Case 6: Using previously defined variables")
    test6_code = """
print(f"Previous stats: {stats}")
"""
    print_result(python_tool.execute({"code": test6_code}))