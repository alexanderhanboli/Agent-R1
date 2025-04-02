"""
Specific tool implementations
"""

from agent_r1.tool.tools.search_tool import SearchTool
from agent_r1.tool.tools.calculator_tool import CalculatorTool
from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
from agent_r1.tool.tools.python_tool import PythonTool
__all__ = [
    'SearchTool',
    'CalculatorTool',
    'WikiSearchTool',
    'PythonTool',
] 

def _default_tools(env):
    """Return a list of tools based on environment type.
    
    Args:
        env: String specifying the environment type or list of tool names
        
    Returns:
        List of tool instances
    """
    # Map of tool names to their classes
    tool_map = {
        'search': SearchTool,
        'calculator': CalculatorTool,
        'wikisearch': WikiSearchTool,
        'python': PythonTool
    }
    
    # If env is a list, return those specific tools
    if isinstance(env, (list, tuple)):
        tools = []
        for tool_name in env:
            if tool_name in tool_map:
                tools.append(tool_map[tool_name]())
        return tools
    
    # Handle single environment types
    if env == 'search':
        return [SearchTool()]
    elif env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    elif env == 'python':
        return [PythonTool()]
    elif env == 'all':
        return [tool_class() for tool_class in tool_map.values()]
    elif env == 'none':
        return []
    else:
        raise NotImplementedError(f"Unknown environment type: {env}")
