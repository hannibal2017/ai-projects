import ollama
response = ollama.chat(model="deepseek-r1", messages=[
    {"role": "user", "content": "如何实现一个 AI 助手？"}
])
print(response['message'])
