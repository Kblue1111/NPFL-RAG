from config.keys import Keys
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, ModelScopeEmbeddings, HuggingFaceBgeEmbeddings
from modelscope import snapshot_download
import os

def get_openaiEmbedding_model():
    return OpenAIEmbeddings(openai_api_key=Keys.OPENAI_API_KEY)

def get_huggingfaceEmbedding_model():
    return HuggingFaceEmbeddings(model_name="/home/kangxiaolan/miniconda3/envs/llmfl/sentence-t5-large")

def get_modelscopeEmbeddings():
    model_dir = "AI-ModelScope/bge-large-en-v1.5"
    model_cache_dir = os.path.join("/home/kangxiaolan/.cache/modelscope/hub", model_dir.replace('.', '___'))

    # 检查模型是否已下载
    if not os.path.exists(model_cache_dir):
        print(f"模型未找到，开始下载：{model_dir}")
        model_cache_dir = snapshot_download(model_dir, revision='master')
    else:
        print(f"模型已存在，直接使用：{model_cache_dir}")

    # 初始化模型
    model_name = model_cache_dir  # 使用下载的模型目录作为模型名称
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}  # 设置为True计算余弦相似度

    # 初始化 HuggingFaceBgeEmbeddings
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="为这个句子生成表示以用于检索相关文章："
    )

    return embeddings

def get_modelscopeEmbeddings_rank():
    model_dir = "BAAI/bge-reranker-large"
    model_cache_dir = os.path.join("/home/kangxiaolan/.cache/modelscope/hub", model_dir.replace('.', '___'))

    # 检查模型是否已下载
    if not os.path.exists(model_cache_dir):
        print(f"模型未找到，开始下载：{model_dir}")
        model_cache_dir = snapshot_download(model_dir, revision='master')
    else:
        print(f"模型已存在，直接使用：{model_cache_dir}")

    # 初始化模型
    model_name = model_cache_dir  # 使用下载的模型目录作为模型名称
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}  # 设置为True计算余弦相似度

    # 初始化 HuggingFaceBgeEmbeddings
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="为这个句子生成表示以用于检索相关文章："
    )

    return embeddings