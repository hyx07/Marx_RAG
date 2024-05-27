# Marx_RAG

安装
1.安装pytorch，见:www.pytorch.org
2.安装第三方库。
  pip install transformers accelerate bitsandbytes langchain sentence-transformers faiss-gpu
3.下载马恩全集faiss数据库，百度网盘：
4.下载embedding模型，reranker模型以及llm：
  embedding: https://huggingface.co/BAAI/bge-large-zh-v1.5
  reranker: https://huggingface.co/BAAI/bge-reranker-v2-m3
  llm: https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat

使用
编辑llama_rag.py，填写实际的embedding模型路径，reranker模型路径，llm模型路径。
运行llama_rag.py即可。

其他
参考faiss_generation.py以及merge_database.py可以制作其他预料的faiss数据库。
