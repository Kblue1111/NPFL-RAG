from langchain_community.retrievers import BM25Retriever
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import jieba
from rank_bm25 import BM25Okapi
import os.path
from models.embedding_model import get_modelscopeEmbeddings, get_modelscopeEmbeddings_rank
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import SendPrompt
from KnowledgeBase.getFunction import extract_json
import json

# 加载排序模型
# tokenizer = AutoTokenizer.from_pretrained('../hugging-face-model/BAAI/bge-reranker-base/')
# rerank_model = AutoModelForSequenceClassification.from_pretrained('../hugging-face-model/BAAI/bge-reranker-base/')
# rerank_model.cuda()

def extract_text(files_loc):
    loader = TextLoader(files_loc)
    text = loader.load()
    return text


def split_content(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
        separators=['\n']
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def save_vectorstore(content_chunks, embedding_model, db_path):
    db = FAISS.from_documents(content_chunks, embedding_model)
    db_location = os.path.join(db_path)
    db.save_local(db_location)
    return db


def preprocessing_func(text: str) -> List[str]:
    return list(jieba.cut(text))


def retrieve(content_chunks, prompts, db):
    texts = [i.page_content for i in content_chunks]
    texts_processed = [preprocessing_func(t) for t in texts]
    vectorizer = BM25Okapi(texts_processed)
    # 文本召回
    bm25_res = vectorizer.get_top_n(preprocessing_func(prompts), texts, n=1)
    # 向量召回
    vector_res = db.similarity_search(prompts, k=1)
    # print("bm25", bm25_res)
    # print("vector", vector_res)
    return bm25_res, vector_res


# 多路召回，加权
def rrf(vector_results: List[str], text_results: List[str], k: int = 1, m: int = 60):
    """
    使用RRF算法对两组检索结果进行重排序

    params:
    vector_results (list): 向量召回的结果列表,每个元素是专利ID
    text_results (list): 文本召回的结果列表,每个元素是专利ID
    k(int): 排序后返回前k个
    m (int): 超参数

    return:
    重排序后的结果列表,每个元素是(文档ID, 融合分数)
    """

    doc_scores = {}

    # 遍历两组结果,计算每个文档的融合分数
    for rank, doc_id in enumerate(vector_results):
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank + m)
    for rank, doc_id in enumerate(text_results):
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank + m)

    # 将结果按融合分数排序
    # sorted_dict = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    sorted_results = [d for d, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]]
    return sorted_results

# 重排序函数
def rerank_with_model(pairs):
    tokenizer = AutoTokenizer.from_pretrained("/home/kangxiaolan/.cache/modelscope/hub/BAAI/bge-reranker-base")
    rerank_model = AutoModelForSequenceClassification.from_pretrained('/home/kangxiaolan/.cache/modelscope/hub/BAAI/bge-reranker-base')
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将模型移动到相应设备
    rerank_model = rerank_model.to(device)
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: value.to(device) for key, value in inputs.items()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

    return scores


def order(bm25_res, vector_res, query):
    vector_results = [i.page_content for i in vector_res]
    text_results = [i for i in bm25_res]

    # vector_RAG_location = os.path.join(res_path, "vector_RAG_1.txt")
    # with open(vector_RAG_location, 'w') as file:
    #     file.write(str(vector_results))
    #
    # text_RAG_location = os.path.join(res_path, "text_RAG_2.txt")
    # with open(text_RAG_location, 'w') as file:
    #     file.write(str(text_results))
    # print("vector", vector_results)
    # print("text", text_results)

    sorted_results = rrf(vector_results, text_results)
    # sorted_results_location = os.path.join(res_path, "sorted_2.txt")
    # with open(sorted_results_location, 'w') as file:
    #     file.write(str(sorted_results))
    #
    # pairs = [[query, result] for result in sorted_results]
    # 获取重排序分数
    # scores = rerank_with_model(pairs)
    # scores_location = os.path.join(res_path, 'score_2.txt')
    # with open(scores_location, 'w') as file:
    #     file.write(str(scores))

    # print("scores", scores)
    # 根据分数排序结果
    # top_indices = scores.cpu().numpy().argsort()[-2:][::-1]  # 取最高两个分数的索引，并反转顺序得到从高到低
    # top_results = [sorted_results[idx] for idx in top_indices]
    # result = sorted_results[scores.cpu().numpy().argmax()]
    # print("sogi", result)
    return sorted_results
    # rrf_res = rrf(vector_results, text_results)
    # print("result", rrf_res)
    # return vector_results

# 生成功能语义
def generate_function(prompt_location1, experiment_model, code):
    # 读取提示词
    with open(prompt_location1, 'r', encoding='utf-8') as file:
        prompt1 = file.read()

    prompt = f"{prompt1}\nBuggy Code:\n{code}\n\n"
    results = SendPrompt.send_prompt_openai_gpt(prompt, experiment_model)
    function = extract_json(results)

    combined_query = f"{code}\nFunctional Semantics: {function}"
    return combined_query

def RAG(files_loc, db_path, res_path, code_location):
    # 加载文档
    text = extract_text(files_loc)
    # 将获取到的数据内容划分
    chunks = split_content(text)
    # 对每个chunk计算embedding，并存入到向量数据库
    embedding_model = get_modelscopeEmbeddings()
    # 创建向量数据库对象，并将文本embedding后存入到里面
    db = save_vectorstore(chunks, embedding_model, db_path)

    # 错误代码
    with open(code_location, 'r', encoding='utf-8') as file:
        code = file.read()
    # 获取功能语义提示词
    # prompt_location1 = "/home/kangxiaolan/miniconda3/envs/llmfl/LLMFL/prompts/function_prompt.txt"
    # experiment_model = "gpt-3.5-turbo"
    # experiment_model = "gpt-4o"

    # 代码 + 功能语义 = query
    # query = generate_function(prompt_location1, experiment_model, code)
    # 向量相似度匹配
    bm25_res, vector_res = retrieve(chunks, code, db)
    # 召回结果排序
    rrf_res = order(bm25_res, vector_res, code)
    return rrf_res

def final_RAG(files_loc, db_path, res_path, code_location):
    # RAG检索结果
    rrf_res = RAG(files_loc, db_path, res_path, code_location)
    # 提取RAG检索结果中的 Fault Causes 和 Fix Solution
    fault_causes = []
    fix_solutions = []

    res_data = json.loads(rrf_res)
    try:
        fault_cause = res_data.get("Fault Causes", "")
        fix_sol = res_data.get("Fix Solution", "")
        if fault_cause:
            fault_causes.append(fault_cause)
        if fix_sol:
            fix_solutions.append(fix_sol)
    except json.JSONDecodeError:
        print("无法解析")

    # 将多个Fault Causes和Fix Solutions合并为单个字符串
    fault_causes_str = "\n".join(fault_causes)
    fix_solutions_str = "\n".join(fix_solutions)

    return fault_causes_str, fix_solutions_str
