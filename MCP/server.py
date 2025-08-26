import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP
from typing import Any, Dict, List, Optional

# --- 1. 读取服务器的配置 ---
def get_server_params(server_name: str):
    """读取配置并为指定服务器创建参数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', '.vscode', 'mcp.json')
    
    with open(config_path, "r", encoding="utf-8") as f:
        mcp_config = json.load(f)

    config = mcp_config["servers"][server_name]
    return StdioServerParameters(
        command=config.get("command"),
        args=config.get("args", []),
        env=config.get("env", {})
    )


def get_all_server_names() -> List[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', '.vscode', 'mcp.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        mcp_config = json.load(f)
    return list(mcp_config.get('servers', {}).keys())

# --- 2. 创建我们自己的主服务器 ---
mcp = FastMCP(name="Main Aggregate Server")

# --- 3. 定义代理工具 ---

# a. 代理 Filesystem Server 的工具
@mcp.tool()
async def list_directory(path: str) -> str:
    """代理工具：列出文件系统中的目录内容。"""
    fs_params = get_server_params("filesystem")
    async with stdio_client(fs_params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool("list_directory", arguments={"path": path})
            return result.content[0].text

@mcp.tool()
async def read_file(path: str) -> str:
    """代理工具：读取文件内容。"""
    fs_params = get_server_params("filesystem")
    async with stdio_client(fs_params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool("read_file", arguments={"path": path})
            return result.content[0].text

# b. 代理 GitHub Server 的工具
@mcp.tool()
async def list_github_issues(owner: str, repo: str) -> str:
    """代理工具：列出指定GitHub仓库的Issues。"""
    github_params = get_server_params("github")
    async with stdio_client(github_params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool("list_issues", arguments={"owner": owner, "repo": repo})
            return result.content[0].text


@mcp.tool()
async def list_remote_tools(server_name: str) -> List[str]:
    """列出某个远端服务器的可用工具名列表。"""
    try:
        params = get_server_params(server_name)
    except Exception as e:
        return [f"error: {e}"]

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [t.name for t in tools.tools]


@mcp.tool()
async def call_remote(server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
    """通用代理：在指定的远端服务器上调用工具并返回其结果内容文本。

    参数:
      server_name: 在 mcp.json 中定义的服务器名称
      tool_name: 远端工具名
      arguments: 可选的字典参数
    返回: 工具返回的文本（如果有）或原始结果对象的字符串化表示
    """
    try:
        params = get_server_params(server_name)
    except Exception as e:
        return f"error: cannot find server '{server_name}': {e}"

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            # 调用远端工具
            try:
                result = await session.call_tool(tool_name, arguments=arguments or {})
            except Exception as e:
                return f"error calling remote tool: {e}"

            # result.content 可能包含多个部分，优先返回文本
            try:
                return result.content[0].text
            except Exception:
                try:
                    return json.dumps(result.content)
                except Exception:
                    return str(result)

# --- 4. 运行我们自己的主服务器 ---
if __name__ == "__main__":
    # 这个主服务器将通过 stdio 运行，等待我们的 agent_client.py 来连接
    mcp.run(transport="stdio")