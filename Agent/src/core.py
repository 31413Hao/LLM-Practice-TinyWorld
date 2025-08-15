from zhipuai import ZhipuAI
from typing import List, Dict, Any
from src.utils import function_to_json
from src.tools import get_current_datetime, add, compare, count_letter_in_string, search_wikipedia, get_current_temperature

SYSTEM_PROMPT = """
你是一个叫python高手的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""

class Agent:
    def __init__(self, client: ZhipuAI, model:str="glm-4",tools:List=[],verbose:bool=True):
        self.client = client
        self.model = model
        self.tools = tools
        self.verbose = verbose
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        """
        获取工具的JSON Schema格式
        """
        return [function_to_json(tool) for tool in self.tools]
    
    # tool_call是大模型返回的调用工具的请求
    def handle_tool_call(self,tool_call):
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        # eval是执行一个字符串表达式的函数
        # **是解包一个字典序为函数参数
        function_call_content = eval(f"{function_name}(**{function_args})")

        return {
            "role":"tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }
    
    # 位于 Agent/src/core.py

    def get_completion(self, prompt: str) -> str:
        # 1. 将用户的最新提问添加到对话历史中
        self.messages.append({"role": "user", "content": prompt})

        # 2. 进入一个循环，直到获得最终的文本回答
        while True:
            # 3. 发起 API 调用
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

            response_message = response.choices[0].message

            # 4. 检查模型是否请求调用工具
            if response_message.tool_calls:
                # a. 如果模型要调用工具，先将其“意图”存入历史
                # 注意：这里的 content 可能是 None
                self.messages.append(
                    {"role": "assistant", "content": response_message.content}
                )
                
                tool_list = []
                # b. 遍历并执行所有工具调用
                for tool_call in response_message.tool_calls:
                    tool_list.append([tool_call.function.name, tool_call.function.arguments])
                    # c. 执行工具并将其结果存入历史
                    tool_response = self.handle_tool_call(tool_call)
                    if "conten" in tool_response:
                        tool_response["content"] = tool_response.pop("conten")
                    self.messages.append(tool_response)
                
                if self.verbose:
                    print("调用工具:", response_message.content, tool_list)
                
                # d. 继续下一次循环，将工具结果提交给模型
                continue
            else:
                # e. 如果没有工具调用，说明模型已给出最终答案
                # 将模型的最终回答添加到历史记录
                self.messages.append(
                    {"role": "assistant", "content": response_message.content}
                )
                # f. 返回最终回答并退出循环
                return response_message.content