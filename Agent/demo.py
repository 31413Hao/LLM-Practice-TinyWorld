from zhipuai import ZhipuAI
from dotenv import load_dotenv
import os
from src.core import Agent
from src.tools import add, count_letter_in_string, compare, get_current_datetime, search_wikipedia, get_current_temperature

load_dotenv()

API_KEY = os.getenv("API_KEY")
model_name = os.getenv("llm_model")

if __name__ == "__main__":
    client = ZhipuAI(api_key  = API_KEY)
    agent = Agent(client=client,model=model_name,tools=[get_current_datetime, add, compare, count_letter_in_string],verbose=True)

    while True:
    # 使用彩色输出区分用户输入和AI回答
        prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
        if prompt == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)  # 绿色显示AI助手回答
