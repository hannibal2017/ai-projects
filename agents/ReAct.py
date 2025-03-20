from dotenv import load_dotenv
from langchain_community.llms import Ollama
load_dotenv()

# 导入LangChain Hub
from langchain import hub
# 从hub中获取React的Prompt
prompt = hub.pull("hwchase17/react")
print(prompt)

#####openAI付费的，改成加载本地 DeepSeek
# 导入ChatOpenAI
from langchain_community.llms import OpenAI
# 选择要使用的LLM
# llm = OpenAI()

# 加载本地 DeepSeek
llm = Ollama(model="deepseek-r1")

# 导入SerpAPIWrapper即工具包
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
# 实例化SerpAPIWrapper
search = SerpAPIWrapper()
# 准备工具列表
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="当大模型没有相关知识时，用于搜索知识"
    ),
]

# # 导入create_react_agent功能
# from langchain.agents import create_react_agent
# # 构建ReAct代理
# agent = create_react_agent(llm, tools, prompt)
#
# # 导入AgentExecutor
# from langchain.agents import AgentExecutor
# # 创建代理执行器并传入代理和工具
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
# 记忆上下文
memory = ConversationBufferMemory(memory_key="chat_history")

# 重新初始化 Agent
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # 使用 ZeroShot 代理
    verbose=True,
    memory=memory
)

# 调用代理执行器，传入输入数据
print("第一次运行的结果：")
agent_executor.invoke({"input": "1+1等于多少?中文输出"})
print("第二次运行的结果：")
agent_executor.invoke({"input": "1*8等于多少?"})