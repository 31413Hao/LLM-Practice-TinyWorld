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
            "conten": function_call_content,
            "tool_call_id": function_id,
        }
    
    # 
    def get_completion(self,prompt)->str:
        self.messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )
        if response.choices[0].message.tool_calls:
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            if self.verbose:
                print("调用工具:", response.choices[0].message.content,tool_list)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content