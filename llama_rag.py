from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from FlagEmbedding import FlagReranker
import torch
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils import *

##
max_length = 6  #保存的历史对话数
knowledge_base_dir = 'marx'  #faiss数据库名字
embedder_model_name = '/home/he/algo/embedder/bge-large-zh-v1.5'  #embedding模型路径
reranker_model_name = '/home/he/algo/embedder/bge-reranker-v2-m3' #排序模型路径
llm_model_path = '/home/he/algo/llama3_chinese/llama3-v2.1'  #llm模型路径
##

embedding_model = HuggingFaceEmbeddings(
    model_name = embedder_model_name,
    multi_process = False,
    model_kwargs = {'device':'cuda'},
    encode_kwargs = {'normalize_embeddings':True},
)

knowledge_database = FAISS.load_local('./database/%s'%knowledge_base_dir, embedding_model, allow_dangerous_deserialization=True)

tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

bnb_config4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    )

bnb_config8 = BitsAndBytesConfig(
    load_in_8bit=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    llm_model_path,
    torch_dtype='auto',
    device_map='auto',
    quantization_config = bnb_config4 #use bnb_config8 if a bigger GPU ram.
)

reranker = FlagReranker(reranker_model_name, use_fp16=True)
#####

pre_text = '''你正在扮演卡尔.马克思。请根据下面提供的马克思的著作内容，以及与用户的历史对话，扮演马克思并回答用户的问题。
回答应简明扼要。部分著作内容中可能插入有注释文字，请忽略这些注释。
如果答案无法从著作内容中推断出来，则不要给出答案。
---
著作内容：
{context}
---
历史对话：
{history}
---
需要回答的用户问题：
{question}'''

print('M_bot: Hi，我的朋友，想聊点什么？')
history = []
while True:
    user_input = input('用户:')
    if user_input == 'new': #输入"new"表示开启新一轮对话。
        history = []
        print('---新的对话---')
        continue
    elif user_input == 'exit': #输入"exit"退出程序
        break
    
    history.append(user_input)
    query1 = user_input

    if len(history) > 1:
        query2 = make_query(history)
        
        retrieved_docs1 = knowledge_database.similarity_search(query=query1, k=20)
        retrieved_docs1 = rerank(reranker, query1, retrieved_docs1, k=5)

        retrieved_docs2 = knowledge_database.similarity_search(query=query2, k=20)
        retrieved_docs2 = rerank(reranker, query2, retrieved_docs2, k=5)
    else:
        retrieved_docs1 = knowledge_database.similarity_search(query=query1, k=20)
        retrieved_docs1 = rerank(reranker, query1, retrieved_docs1, k=10)

        retrieved_docs2 = []

    retrieved_docs = retrieved_docs1 + retrieved_docs2
    
    context = ''
    context += "".join([f"资料 {str(i)}:\n" + doc.page_content + '\n***\n' for i, doc in enumerate(retrieved_docs)])

    conversation_history = make_history(history[:-1])
    if conversation_history is None:
        conversation_history = '无\n'
        
    final_prompt = pre_text.format(context=context, history=conversation_history, question=user_input)
    #uncomment to see the actual prompt, segments extracted, etc.
    #print(final_prompt)

    messages = [
    {"role": "user", "content": final_prompt},]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    history.append(response)
    if len(history) > max_length:
        history = history[-max_length:]
    print('M_bot:'+response)
    
