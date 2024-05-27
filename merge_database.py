from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
import os

dir_files = os.listdir('./database/')

model_name = '/home/he/algo/embedder/bge-large-zh-v1.5'
embedding_model = HuggingFaceEmbeddings(
    model_name = model_name,
    multi_process = False,
    model_kwargs = {'device':'cuda'},
    encode_kwargs = {'normalize_embeddings':True},
)

knowledge_database = FAISS.load_local('./database/'+dir_files[0], embedding_model, allow_dangerous_deserialization=True)

dir_list = dir_files[1:]
for dir_path in dir_list:
    print('merging '+dir_path)
    current_base = FAISS.load_local('./database/%s'%dir_path, embedding_model, allow_dangerous_deserialization=True)
    knowledge_database.merge_from(current_base)

knowledge_database.save_local('marx')
