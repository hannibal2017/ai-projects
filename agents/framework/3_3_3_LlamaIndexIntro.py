# 读取系统变量
from dotenv import load_dotenv
load_dotenv()
import os

# 从指定目录读取文档数据
from llama_index.core import SimpleDirectoryReader
# 获取当前项目的根目录
project_root = os.getcwd()
print("projectroot:",project_root)

# 拼接 data 目录的路径
data_path = os.path.join(project_root, "framework/data")
documents = SimpleDirectoryReader(data_path).load_data()

# # 使用读取到的文档数据创建向量存储索引
# from llama_index.core import VectorStoreIndex
# index = VectorStoreIndex.from_documents(documents)
#
# # 将索引转换为查询引擎Agent
# agent = index.as_query_engine()

# 2️⃣ 替换 OpenAI，使用本地 Ollama 部署的 DeepSeek
from langchain_community.llms import Ollama
llm = Ollama(model="deepseek-r1")

# 3️⃣ 替换 OpenAI 的嵌入模型，使用 `huggingface`（本地）或者 `llama-cpp-python`
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")  # 替换为适合你的模型

# 4️⃣ 使用本地 LLM 和本地 Embedding 创建索引
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# 5️⃣ 使用本地 LLM 进行查询
agent = index.as_query_engine(llm=llm)


# 查询并打印结果
response = agent.query("花语秘境的员工有几处角色?")
print("花语秘境的员工有几处角色?", response)
response = agent.query("花语秘境的Agent叫啥名字")
print("花语秘境的Agent叫啥名字",response)

# 将索引的存储上下文持久化
index.storage_context.persist()