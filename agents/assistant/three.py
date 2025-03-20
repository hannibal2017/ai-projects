from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 运行本地 DeepSeek
llm = Ollama(model="deepseek-r1")

# 自定义 Prompt
prompt = PromptTemplate.from_template("请回答以下问题：{question}")

# 让 AI 进行推理
chain = LLMChain(llm=llm, prompt=prompt)

response = chain.invoke({"question": "Python 如何实现二分查找？"})
print(response)
