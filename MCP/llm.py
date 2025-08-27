import fitz
import os
import re
import json
import numpy as np
from tqdm import tqdm
from zhipuai import ZhipuAI
from dotenv import load_dotenv

load_dotenv()

issues_dir = "issues"
json_name = "servers"
json_path = os.path.join(issues_dir,json_name+".json")
#json_path = "issues/server.json"

API_KEY = os.getenv("API_KEY")
client = ZhipuAI(api_key=API_KEY)

llm_model = os.getenv("llm_model")
embedding_model = os.getenv("embedding_model")

def extract_text_from_json(json_path):
    """
    从JSON文件中提取并组合所有issue的标题和正文作为知识库文本。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        issues_data = json.load(f)
    
    all_text = ""
    for issue in issues_data:
        # 提取标题，如果不存在则为空字符串
        title = issue.get('title', '')
        # 提取正文，如果不存在则为空字符串
        body = issue.get('body', '')
        
        # 将标题和正文拼接起来，每个issue之间用分隔符隔开
        all_text += f"Issue Title: {title}\n\n"
        if body: # 确保body不为None
            all_text += f"Issue Body:\n{body}\n"
        all_text += "---\n\n"
        
    return all_text

def chunk_text(text, n, overlap):
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]
        if chunk:
            chunks.append(chunk)
    return chunks

class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, top_k=5):
        if not self.vectors:
            return []
        
        query_vector = np.array(query_embedding)
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i in range(min(top_k, len(similarities))):
            index, similarity = similarities[i]
            results.append({
                "text": self.texts[index],
                "similarity": similarity,
                "metadata": self.metadata[index]
            })
        return results
    
def create_embeddings(texts: list, batch_size=32):
    """
    使用Embedding模型为给定的文本列表分批创建嵌入向量。

    Args:
        texts (List[str]): 要创建嵌入向量的输入文本列表。
        batch_size (int): 每批处理的文本数量，应小于等于API限制（64）。

    Returns:
        List[List[float]]: 包含所有文本嵌入向量的列表。
    """
    all_embeddings = []
    
    # 将文本列表分割成多个批次
    for i in tqdm(range(0, len(texts), batch_size), desc="正在创建嵌入向量 (分批)"):
        batch = texts[i:i + batch_size]
        
        response = client.embeddings.create(
            model=embedding_model,
            input=batch
        )
        
        # 将当前批次的结果添加到总列表中
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        
    return all_embeddings

def process_knowledge_base(file_path, chunk_size=1000, chunk_overlap=200):
    """
    为RAG处理知识库文件（现在是JSON）。
    """
    print("从JSON知识库提取文本...")
    text = extract_text_from_json(file_path)

    print("分割文本块...")
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    print(f"总共分割为 {len(chunks)} 个文本块。")

    print("创建嵌入向量...")
    chunk_embeddings = create_embeddings(chunks)
    
    vector_store = SimpleVectorStore()
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        metadata = {
            "index": i,
            "source": file_path
        }
        vector_store.add_item(chunk, embedding, metadata)
    
    print(f"向向量存储中添加了 {len(chunks)} 个文本块")
    return vector_store

def compress_chunk(chunk, query, compression_type="selective"):
    # ... (此函数无需修改)
    if compression_type == "selective":
        system_prompt = "您是专业信息过滤专家..."
    elif compression_type == "summary":
        system_prompt = "您是专业摘要生成专家..."
    else: # extraction
        system_prompt = "您是精准信息提取专家..."
    
    user_prompt = f"查询: {query}\n\n文档块:\n{chunk}\n\n请严格提取与本查询相关的文档块的核心内容。"
    
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    
    compressed_chunk = response.choices[0].message.content.strip()
    original_length = len(chunk)
    compressed_length = len(compressed_chunk)
    compression_ratio = (original_length - compressed_length) / original_length * 100 if original_length > 0 else 0
    
    return compressed_chunk, compression_ratio

def batch_compress_chunks(chunks, query, compression_type="selective"):
    # ... (此函数无需修改)
    print(f"正在压缩 {len(chunks)} 个文本块...")
    results = []
    sum_original_length = 0
    sum_compressed_length = 0
    for i, chunk in enumerate(tqdm(chunks, desc="压缩进度")):
        compressed_chunk, compression_ratio = compress_chunk(chunk, query, compression_type)
        results.append((compressed_chunk, compression_ratio))
        sum_original_length += len(chunk)
        sum_compressed_length += len(compressed_chunk)
    
    total_compression_ratio = (sum_original_length - sum_compressed_length) / sum_original_length * 100 if sum_original_length > 0 else 0
    print(f"总体压缩比率: {total_compression_ratio:.2f}%")
    
    return results

def generate_response(query, context):
    # ... (此函数无需修改)
    system_prompt = "您是一个乐于助人的AI助手..."
    user_prompt = f"上下文:\n{context}\n\n问题: {query}\n\n请基于上述上下文内容提供一个全面详尽的答案。"
    
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content

def rag_with_compression(file_path, query, k=10, compression_type="selective"):
    """
    完整的RAG管道，包含上下文压缩。
    """
    print("\n=== RAG WITH CONTEXTUAL COMPRESSION ===")
    print(f"Query: {query}")
    print(f"Compression type: {compression_type}")

    # 1. 处理知识库 (JSON)
    vector_store = process_knowledge_base(file_path)

    # ... (后续流程无需修改)
    query_embedding = create_embeddings(query)
    print(f"Retrieving top {k} chunks...")
    results = vector_store.similarity_search(query_embedding, top_k=k)
    retrieved_chunks = [result["text"] for result in results]

    compressed_results = batch_compress_chunks(retrieved_chunks, query, compression_type)
    compressed_chunks = [result[0] for result in compressed_results]
    compression_ratios = [result[1] for result in compressed_results]

    filtered_chunks = [(chunk, ratio) for chunk, ratio in zip(compressed_chunks, compression_ratios) if chunk.strip()]
    if not filtered_chunks:
        print("所有块都被压缩为空，使用原始块。")
        filtered_chunks = [(chunk, 0) for chunk in retrieved_chunks]
    
    compressed_chunks, compression_ratios = zip(*filtered_chunks)
    context = "\n\n".join(compressed_chunks)
    response = generate_response(query, context)
    
    avg_compression_ratio = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0
    result = {
        "query": query,
        "original_retrieved_chunks": retrieved_chunks,
        "compressed_chunks": list(compressed_chunks),
        "compressed_ratios": list(compression_ratios),
        "context_length_reduction": f"{avg_compression_ratio:.2f}%",
        "response": response
    }
    
    print("\n=== RAG RESULT ===")
    print(f"Response: {response}")
    return result

# --- 主程序入口 ---
if __name__ == "__main__":
    query = "用100个字总结一下mcp开发会遇到的问题。"
    # 确保 issues/server.json 文件存在
    # if not os.path.exists(json_path):
    #     print(f"错误: 知识库文件未找到，请确保 '{json_path}' 文件存在。")
    # else:
    #     result = rag_with_compression(json_path, query, k=10, compression_type="selective")
    # # 原生模型回复
    system_prompt = "您是一个乐于助人的AI助手..."
    user_prompt = f"问题: {query}\n\n请基于上述上下文内容提供一个全面详尽的答案。"
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content":system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    print("\n=== ORIGINAL LLM RESPONSE ===")
    print(f"Query: {query}")
    print(response.choices[0].message.content)

    