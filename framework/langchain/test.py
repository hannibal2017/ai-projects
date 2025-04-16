from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 加载本地 DeepSeek
llm = Ollama(model="deepseek-r1")

# 创建 Prompt
prompt = PromptTemplate.from_template("1 + 1等于多少：{question}")

# 创建 LangChain 执行链
chain = LLMChain(llm=llm, prompt=prompt)

# 运行 AI 生成代码
response = chain.invoke({"question": ""})
print(response)

# 提取文本内容
formatted_text = response['text'] if isinstance(response, dict) and 'text' in response else str(response)

# 格式化输出
print("\n" + "="*30)
print("💡 AI 计算结果：\n")
print(formatted_text)
print("="*30 + "\n")