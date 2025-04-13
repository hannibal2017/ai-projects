##rag测试
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_community.llms import Ollama
import gradio as gr

# 1. 替换 OpenAI 嵌入模型，使用 Hugging Face
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. 读取本地文档
documents = SimpleDirectoryReader("./data").load_data()

# 3. 创建知识库索引，使用 Hugging Face 嵌入
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# 4. 绑定本地 DeepSeek
llm = Ollama(model="deepseek-r1")
query_engine = index.as_query_engine(llm=llm)

# # 5. 进行查询
# response = query_engine.query("公司的休假制度是什么？")
# print(response)

# 定义回答函数
def rag_chat(query):
    response = query_engine.query(query)
    return str(response)

# 启动 Gradio 界面
iface = gr.Interface(
    fn=rag_chat,
    inputs=gr.Textbox(lines=2, placeholder="请输入问题...", label="提问"),
    outputs="text",
    title="RAG 问答机器人",
    description="基于本地 DeepSeek 的知识库问答系统"
)

iface.launch()