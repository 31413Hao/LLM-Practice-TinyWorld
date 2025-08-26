import asyncio
import json
import os
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any


def make_local_server_params():
    """创建运行本地聚合服务器（server.py）的 StdioServerParameters。

    这个 agent 将启动并通过 stdio 与本地的 `server.py` 进程通信，
    server.py 会代理 `.vscode/mcp.json` 中列出的后端服务器工具。
    """
    script_dir = Path(__file__).parent
    server_py = script_dir / 'server.py'
    if not server_py.exists():
        raise FileNotFoundError(f"找不到 server.py: {server_py}")

    # 使用当前 Python 解释器启动 server.py
    return StdioServerParameters(command=sys.executable, args=[str(server_py)], env=os.environ.copy())


async def main():
    params = make_local_server_params()

    print("启动并连接本地聚合服务器（server.py）...")
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            print("初始化 server 会话...")
            await session.initialize()
            print("已连接到本地聚合服务器。可用工具:")
            tools = await session.list_tools()
            print([t.name for t in tools.tools])

            # 使用聚合服务器提供的通用代理工具来调用后端服务
            print("\n=== 使用聚合服务器的通用工具 ===")

            # 列出 server.json 中定义的所有远端服务器
            try:
                servers_list_res = await session.call_tool('list_remote_tools', arguments={'server_name': 'filesystem'})
                # 从 CallToolResult 中提取文本
                try:
                    servers_list_text = servers_list_res.content[0].text
                except Exception:
                    servers_list_text = str(servers_list_res)
                print('filesystem 可用工具:', servers_list_text)
            except Exception as e:
                print('list_remote_tools 调用失败:', e)

            # 通过 call_remote 调用远端工具（示例：filesystem 的 list_files）
            try:
                print("\n通过 call_remote 调用 filesystem 的 list_files (path=workspace) ...")
                res = await session.call_tool('call_remote', arguments={'server_name': 'filesystem', 'tool_name': 'list_files', 'arguments': {'path': 'workspace'}})
                print('call_remote result (list_files):', res)
            except Exception as e:
                print('call_remote 调用失败:', e)

            # 示例：调用 GitHub 的代理工具
            try:
                print("\n通过 call_remote 调用 github 的 list_issues ...")
                owner = 'modelcontextprotocol'
                repo = 'servers'
                res_obj = await session.call_tool('call_remote', arguments={'server_name': 'github', 'tool_name': 'list_issues', 'arguments': {'owner': owner, 'repo': repo}})
                # 提取文本内容（优先使用 content[0].text）
                try:
                    text = res_obj.content[0].text
                except Exception:
                    try:
                        text = str(res_obj)
                    except Exception:
                        text = ''

                issues_dir = Path(__file__).parent / 'issues'
                issues_dir.mkdir(parents=True, exist_ok=True)
                out_path_json = issues_dir / f"{repo}.json"
                out_path_txt = issues_dir / f"{repo}.txt"

                saved = False
                if text:
                    # 尝试解析为 JSON
                    try:
                        data = json.loads(text)
                        with open(out_path_json, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        print(f'已将 GitHub issues 保存为 JSON: {out_path_json}')
                        saved = True
                    except Exception:
                        # 尝试截取 JSON 子串
                        try:
                            start = text.find('{')
                            if start == -1:
                                start = text.find('[')
                            if start != -1:
                                trimmed = text[start:]
                                data = json.loads(trimmed)
                                with open(out_path_json, 'w', encoding='utf-8') as f:
                                    json.dump(data, f, ensure_ascii=False, indent=2)
                                print(f'已将 GitHub issues（从文本中提取）保存为 JSON: {out_path_json}')
                                saved = True
                        except Exception:
                            saved = False

                if not saved:
                    # 保存原始文本为 txt
                    with open(out_path_txt, 'w', encoding='utf-8') as f:
                        f.write(text if text else repr(res_obj))
                    print(f'返回内容无法解析为 JSON，已将原始文本保存为 TXT: {out_path_txt}')
            except Exception as e:
                print('call_remote(github) 调用失败:', e)


if __name__ == '__main__':
    asyncio.run(main())