import pytest
from agent_r1.tool import ToolEnv
from agent_r1.tool.tools import PythonTool
from agent_r1.tool.tool_env import step

def test_python_tool_independence():
    """Test that copied environments have independent PythonTool instances"""
    
    # Create initial environment with PythonTool
    env = ToolEnv(tools=[PythonTool()], max_turns=10)
    
    # Create two copies
    env_copy1 = env.copy()
    env_copy2 = env.copy()
    
    # Define a variable in first environment
    action_text1 = '<tool_call>{"name": "python", "arguments": {"code": "x = 42"}}</tool_call>'
    result1, _, _, _ = step(env, action_text1)
    
    # Define a different value for same variable in second environment
    action_text2 = '<tool_call>{"name": "python", "arguments": {"code": "x = 100"}}</tool_call>'
    result2, _, _, _ = step(env_copy1, action_text2)
    
    # Define yet another value in third environment
    action_text3 = '<tool_call>{"name": "python", "arguments": {"code": "x = 200"}}</tool_call>'
    result3, _, _, _ = step(env_copy2, action_text3)
    
    # Check values in each environment
    check_text = '<tool_call>{"name": "python", "arguments": {"code": "print(x)"}}</tool_call>'
    
    result1, _, _, _ = step(env, check_text)
    result2, _, _, _ = step(env_copy1, check_text)
    result3, _, _, _ = step(env_copy2, check_text)
    
    # Parse the results to get the printed values
    def get_value(result):
        import json
        result_dict = json.loads(result)
        return int(result_dict["stdout"].strip())
    
    # Verify each environment has its own independent value
    assert get_value(result1) == 42, "Original environment should have x = 42"
    assert get_value(result2) == 100, "First copy should have x = 100"
    assert get_value(result3) == 200, "Second copy should have x = 200" 