"""
Generic tool environment implementation, usable with any set of tools
"""

import re
import json
import random
import traceback
from typing import Dict, List, Any, Tuple, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy

from agent_r1.tool.tool_base import Tool

# Independent step function
def step(env: 'ToolEnv', action_text: str):
    """
    Execute one step of environment interaction
    
    Args:
        env: The tool environment
        action_text: Text generated by LLM
        
    Returns:
        (observation, reward, done, info)
    """
    env.steps_taken += 1
    action = env.extract_tool_call(action_text)
    
    if action == env.INVALID_ACTION:
        result = "Invalid tool call format. Please use <tool_call>{\"name\": \"tool_name\", \"arguments\": {params_json}}</tool_call> format."
        reward = env.PENALTY_FOR_INVALID  # Syntax error penalty
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=False,
            action_is_effective=False,
            reward=reward
        )
        return result, reward, False, {"action_is_valid": False, "action_is_effective": False}
    
    tool_name = action["tool"]
    tool_args = action["args"]
    
    # Validate if the tool exists
    if tool_name not in env.tool_map:
        result = f"Unknown tool: {tool_name}"
        reward = env.PENALTY_FOR_INEFFECTIVE  # Semantic error penalty
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=False,
            reward=reward
        )
        return result, reward, False, {"action_is_valid": True, "action_is_effective": False}
    
    # Get tool instance
    tool = env.tool_map[tool_name]
    
    # Validate tool arguments
    is_valid, error_msg = tool.validate_args(tool_args)
    if not is_valid:
        result = f"Invalid arguments for tool '{tool_name}': {error_msg}"
        reward = env.PENALTY_FOR_INEFFECTIVE  # Semantic error penalty
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=False,
            reward=reward
        )
        return result, reward, False, {"action_is_valid": True, "action_is_effective": False}
    
    # Execute tool
    try:
        result = tool.execute(tool_args)
        reward = tool.calculate_reward(tool_args, result)
        
        # Record tool call history
        env.tool_history.append({
            "tool": tool_name,
            "args": tool_args,
            "result": result
        })
        
        # Check if max turns reached
        done = env.steps_taken >= env.max_turns
        
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=True,
            reward=reward
        )
        
        return result, reward, done, {"action_is_valid": True, "action_is_effective": True}
    except Exception as e:
        error_trace = traceback.format_exc()
        result = f"Error executing tool '{tool_name}': {str(e)}"
        reward = env.PENALTY_FOR_INEFFECTIVE  # Runtime error penalty
        
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=False,
            reward=reward
        )
        
        return result, reward, False, {"action_is_valid": True, "action_is_effective": False}

# Batch step function
def step_batch(envs: List['ToolEnv'], action_texts: List[str]):
    """
    Execute batch steps of environment interaction
    
    Args:
        envs: List of tool environments
        action_texts: List of texts generated by LLM
        
    Returns:
        List of (observation, reward, done, info) tuples
    """
    assert len(envs) == len(action_texts), "Number of environments and actions must match"
    
    # Group actions by tool name and environment
    tool_groups = {}
    tool_indices = {}
    env_indices = {}
    action_map = {}
    results = [None] * len(envs)
    
    # First pass: extract tool calls and group by tool name
    for i, (env, action_text) in enumerate(zip(envs, action_texts)):
        # Extract the tool call
        action = env.extract_tool_call(action_text)
        action_map[i] = (env, action, action_text)
        
        # Handle invalid actions
        if action == env.INVALID_ACTION:
            result = "Invalid tool call format. Please use <tool_call>{\"name\": \"tool_name\", \"arguments\": {params_json}}</tool_call> format."
            env.steps_taken += 1
            env._update_tracking_variables(
                response=action_text,
                action=action,
                action_is_valid=False,
                action_is_effective=False,
                reward=env.PENALTY_FOR_INVALID  # Syntax error penalty
            )
            results[i] = (result, env.PENALTY_FOR_INVALID, False, {"action_is_valid": False, "action_is_effective": False})
            continue
            
        tool_name = action["tool"]
        tool_args = action["args"]
        
        # Handle unknown tools
        if tool_name not in env.tool_map:
            result = f"Unknown tool: {tool_name}"
            env.steps_taken += 1
            env._update_tracking_variables(
                response=action_text,
                action=action,
                action_is_valid=True,
                action_is_effective=False,
                reward=env.PENALTY_FOR_INEFFECTIVE  # Semantic error penalty
            )
            results[i] = (result, env.PENALTY_FOR_INEFFECTIVE, False, {"action_is_valid": True, "action_is_effective": False})
            print(f"[WARNING] Unknown tool: {result}")
            continue
            
        # Get tool instance
        tool = env.tool_map[tool_name]
        
        # Validate tool arguments
        is_valid, error_msg = tool.validate_args(tool_args)
        if not is_valid:
            result = f"Invalid arguments for tool '{tool_name}': {error_msg}"
            env.steps_taken += 1
            env._update_tracking_variables(
                response=action_text,
                action=action,
                action_is_valid=True,
                action_is_effective=False,
                reward=env.PENALTY_FOR_INEFFECTIVE  # Semantic error penalty
            )
            results[i] = (result, env.PENALTY_FOR_INEFFECTIVE, False, {"action_is_valid": True, "action_is_effective": False})
            print(f"[WARNING] Invalid arguments for tool: {result}")
            continue
            
        # Group by tool name
        if tool_name not in tool_groups:
            tool_groups[tool_name] = []
            tool_indices[tool_name] = []
            env_indices[tool_name] = []
            
        tool_groups[tool_name].append(tool_args)
        tool_indices[tool_name].append(i)
        env_indices[tool_name].append(env)
    
    # Second pass: execute tools in batch where possible
    for tool_name, args_list in tool_groups.items():
        indices = tool_indices[tool_name]
        envs_list = env_indices[tool_name]
        
        # All environments share the same tool instances, so we can use the first one
        tool = envs_list[0].tool_map[tool_name]
        
        # try:
        # Try batch execution
        batch_results = tool.batch_execute(args_list)
        # print(f"[DEBUG] batch_results: {batch_results}")
        
        # Process results
        for idx, env, result, args in zip(indices, envs_list, batch_results, args_list):
            env.steps_taken += 1
            reward = tool.calculate_reward(args, result)
            
            # Record tool call history
            env.tool_history.append({
                "tool": tool_name,
                "args": args,
                "result": result
            })
            
            # Check if max turns reached
            done = env.steps_taken >= env.max_turns
            
            # Update tracking variables
            action_text = action_texts[idx]
            action = action_map[idx][1]
            env._update_tracking_variables(
                response=action_text,
                action=action,
                action_is_valid=True,
                action_is_effective=True,
                reward=reward
            )
            
            results[idx] = (result, reward, done, {"action_is_valid": True, "action_is_effective": True})
    return results

class ToolEnv:
    """
    Generic tool environment class, handling tool calls, history tracking, and state
    
    Action States:
    - action_is_valid: Indicates if the action format is correct (proper JSON format, correct tool call syntax)
                      This is about the syntactic correctness of the action.
    - action_is_effective: Indicates if the action was successfully executed (tool exists, valid args, no runtime errors)
                         This is about the semantic correctness and successful execution of the action.
    
    Examples:
    1. Invalid JSON format: valid=False, effective=False
    2. Unknown tool name: valid=True, effective=False
    3. Invalid tool args: valid=True, effective=False
    4. Runtime error: valid=True, effective=False
    5. Successful execution: valid=True, effective=True
    
    Penalties:
    - PENALTY_FOR_INVALID: Applied when action format is invalid (syntax error)
    - PENALTY_FOR_INEFFECTIVE: Applied when action is valid but fails to execute (semantic/runtime error)
    """
    INVALID_ACTION = {"tool": "invalid", "args": {}}
    PENALTY_FOR_INVALID = -0.1  # Penalty for syntax errors
    PENALTY_FOR_INEFFECTIVE = -0.05  # Penalty for semantic/runtime errors
    
    def __init__(self, tools: List[Tool] = None, max_turns: int = 10):
        """
        Initialize the tool environment
        
        Args:
            tools: List of available tools
            max_turns: Maximum number of interaction turns
        """
        self.tools = tools or []
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.tool_desc = [tool.get_description() for tool in self.tools]
        self.max_turns = max_turns
        self.reset_tracking_variables()

    def tools_format_func(self) -> str:
        template = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
        tools = "\n".join([f"{json.dumps(tool.get_description(), ensure_ascii=False)}" for tool in self.tools])
        return template.format(tools=tools)
        
    def reset_tracking_variables(self):
        """Reset tracking variables"""
        self.rewards = []  # Record rewards for each tool call
        self.tool_history = []  # Record tool call history
        self.steps_taken = 0
        self._actions = []  # All actions (including all LLM responses)
        self._actions_valid = []  # Correctly formatted actions
        self._actions_effective = []  # Effectively executed actions
    
    def get_tracking_variables(self) -> Dict:
        """Get statistics of tracking variables"""
        return {
            "rewards": self.rewards,  # List of rewards for each step
            "total_reward": sum(self.rewards),  # Total reward for convenience
            "steps_taken": self.steps_taken,
            "tool_history": self.tool_history,
            "actions": self._actions,
            "actions_valid": self._actions_valid,
            "actions_effective": self._actions_effective,
        }
    
    def _update_tracking_variables(
            self, 
            response: str,
            action: Any, 
            action_is_valid: bool,
            action_is_effective: bool,
            reward: float,
        ):
        """
        Update tracking variables
        
        Args:
            response: Raw LLM response
            action: Parsed action
            action_is_valid: Whether the action format is syntactically valid (proper JSON, correct tool call syntax)
            action_is_effective: Whether the action was semantically valid and successfully executed
            reward: Reward for the current step (including any penalties)
            
        Note:
            - action_is_valid=True means the action was properly formatted, but doesn't guarantee it can be executed
            - action_is_effective=True means the action was both valid and successfully executed
            - action_is_effective can only be True if action_is_valid is True
        """
        self._actions.append(response)
        if action_is_valid:
            self._actions_valid.append(action)
        else:
            self._actions_valid.append(None)
        if action_is_effective:
            self._actions_effective.append(action)
        else:
            self._actions_effective.append(None)
        
        self.rewards.append(reward)
    
    def extract_tool_call(self, text: str) -> Dict:
        """
        Extract tool call from LLM output
        
        Args:
            text: Text generated by LLM
            
        Returns:
            Dictionary containing tool name and parameters
        """
        # Regular expression to extract tool call <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        
        tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)
        
        if not tool_call_match:
            return self.INVALID_ACTION
        
        try:
            tool_call_json = tool_call_match.group(1).strip()
            # Handle potentially non-standard JSON format
            # tool_call_json = tool_call_json.replace("'", '"')
            tool_call_data = json.loads(tool_call_json)
            
            # Extract tool name and arguments
            if "name" not in tool_call_data:
                return self.INVALID_ACTION
                
            tool_name = tool_call_data["name"]
            tool_args = tool_call_data.get("arguments", {})
            
            return {"tool": tool_name, "args": tool_args}
        except json.JSONDecodeError:
            return self.INVALID_ACTION
        except Exception:
            return self.INVALID_ACTION
    
    def get_tool_history_context(self) -> str:
        """
        Generate tool call history context
        
        Returns:
            Formatted tool call history
        """
        if not self.tool_history:
            return "No tool call history yet."
        
        context = "Tool call history:\n"
        for i, call in enumerate(self.tool_history):
            context += f"{i+1}. Tool: {call['tool']}\n"
            context += f"   Arguments: {json.dumps(call['args'], ensure_ascii=False)}\n"
            context += f"   Result: {call['result']}\n\n"
        
        return context
    
    def get_available_tools_description(self) -> str:
        """
        Get description of available tools
        
        Returns:
            Formatted tool descriptions
        """
        if not self.tools:
            return "No tools available."
            
        descriptions = ["Available tools:"]
        for tool in self.tools:
            descriptions.append(tool.get_simple_description())
            
        return "\n\n".join(descriptions)
    
    def copy(self):
        """
        Copy the tool environment
        
        Creates new instances of stateful tools (like PythonTool) while reusing stateless tools
        """
        # Create new instances of stateful tools, reuse stateless ones
        new_tools = []
        for tool in self.tools:
            if tool.__class__.__name__ == 'PythonTool':
                # Create a fresh instance of PythonTool
                # Python code execution in one environment won't affect the state of Python variables in other environments
                new_tools.append(tool.__class__())
            else:
                # Reuse stateless tools
                new_tools.append(tool)
                
        env = ToolEnv(tools=new_tools, max_turns=self.max_turns)
        env.tool_history = deepcopy(self.tool_history)
        env.rewards = deepcopy(self.rewards)
        env.steps_taken = self.steps_taken
        env._actions = deepcopy(self._actions)
        env._actions_valid = deepcopy(self._actions_valid)
        env._actions_effective = deepcopy(self._actions_effective)
        return env