import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel
from datasets import load_dataset
import re
import torch
import numpy as np
import faiss
import torch.nn.functional as F
from typing import List, Tuple
import pdb
from image_worker import find_same_features


from test_text_mask import find_most_similar_sentences

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的前端域名，生产环境建议设置为具体的域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许的 HTTP 方法
    allow_headers=["*"],  # 允许的请求头
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'BAAI/bge-m3'
model = AutoModel.from_pretrained(model_name)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name,)
batch_size = 32
sep_token = "<|SEP|>"


# 定义请求数据模型
class TextPrompt(BaseModel):
    target: str
    paragraph: str


@app.post("/compute")
async def compute_text(request: TextPrompt):
    target = request.target
    paragraph = request.paragraph

    # 计算相似度
    results = find_most_similar_sentences(
        target_sentence=target,
        paragraph=paragraph,
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=3,
        min_length=10  # 设置最小句子长度
    )

    # 打印结果
    print(f"目标句子: {target}\n")
    print("最相似的句子:")
    for sentence, score in results:
        print(f"相似度 {score:.4f}: {sentence}")

    return {"result": results}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8123)