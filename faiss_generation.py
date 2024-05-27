import datasets
from langchain.docstore.document import Document as LangchainDocument
from tqdm import tqdm
from typing import Optional, List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

source_name = '第五十卷'

embedder_model_name = '/home/he/algo/embedder/bge-large-zh-v1.5' #embedding模型路径
dataset = datasets.load_dataset('text', data_dir='/home/he/datasets/marx1', sample_by='document', split='train')
Raw_knowledge_base = [
    LangchainDocument(page_content=doc['text']) for doc in tqdm(dataset)
    ]

SEPARATORS = [
        "\u3002",  #中文句号作为分割符号
]

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name) -> List[LangchainDocument]:
    
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size/10.0),
        add_start_index=True,
        strip_whitespace=True,
        keep_separator=False,
        separators=SEPARATORS,
        )

    docs_processed = []
    for doc in Raw_knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            doc.page_content += '。'
            doc.metadata['source'] = '马恩全集'+source_name
            docs_processed_unique.append(doc)

    return docs_processed_unique

print('begin docs process!')
docs_processed = split_documents(
    256,
    Raw_knowledge_base,
    tokenizer_name = embedder_model_name)

print('docs processed! length:', len(docs_processed))
print(len(docs_processed))

embedding_model = HuggingFaceEmbeddings(
    model_name = embedder_model_name,
    multi_process = False,
    model_kwargs = {'device':'cuda'},
    encode_kwargs = {'normalize_embeddings':True},
)

print('begin embed!')

db = None
with tqdm(total=len(docs_processed),desc='Ingesting documents') as pbar:
    for d in docs_processed:
        if db:
            db.add_documents([d])
        else:
            db = FAISS.from_documents([d], embedding_model, distance_strategy=DistanceStrategy.COSINE)
        pbar.update(1)

db.save_local('./database/'+source_name)

























