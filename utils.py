import numpy as np

def rerank(reranker, question, context, k):
    scores = []
    for text in context:
        current_score = reranker.compute_score([question, text.page_content])
        scores.append(current_score)

    sort = np.argsort(-np.array(scores))
    new_context = []
    for i in range(k):
        new_context.append(context[sort[i]])

    return new_context

def make_query(history):
    query = ''
    for i in range(len(history)):
        if i%2 == 0:
            query = query + '问：' + history[i] + '\n'
        else:
            query = query + '答：' + history[i] + '\n'
    return query

def make_history(history):
    if len(history) < 1:
        return None
    
    conversation = ''
    for i in range(len(history)):
        if i%2 == 0:
            conversation = conversation + '用户：' + history[i] + '\n'
        else:
            conversation = conversation + '你：' + history[i] + '\n'
    return conversation
