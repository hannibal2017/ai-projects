from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 加载本地 Embedding
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")

# 读取文档并向量化
documents = ["OpenAI Assistant 提供 API，但收费...", "DeepSeek 是一个不错的替代品"]
faiss_db = FAISS.from_texts(documents, embedding_model)
retriever = faiss_db.as_retriever()

# 查询 AI 助手
query = "有什么开源 AI 助手方案？"
docs = retriever.get_relevant_documents(query)
print(docs)
