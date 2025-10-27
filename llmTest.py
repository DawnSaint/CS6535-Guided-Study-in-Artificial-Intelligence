# from websocietysimulator import Simulator
from websocietysimulator.llm import LLMBase, InfinigenceLLM

# simulator = Simulator(data_dir="", device="gpu", cache=True)
# simulator.set_llm(InfinigenceLLM(api_key="sk-f3obch3f7dfxlg4l"))


llm = InfinigenceLLM(api_key="sk-f3obch3f7dfxlg4l")

messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "你好，帮我介绍一下人工智能。"}
]

response = llm(messages)
print(repr(response))