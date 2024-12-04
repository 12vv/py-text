import os
# os.environ['HF_HOME'] = '/mnt/A/huggingface_cache'
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import re
import torch
import numpy as np
import faiss
import torch.nn.functional as F
from typing import List, Tuple
import pdb
from image_worker import find_same_features


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'BAAI/bge-m3'
model = AutoModel.from_pretrained(model_name)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name,)
batch_size = 32
sep_token = "<|SEP|>"

ds = load_dataset("Anthropic/hh-rlhf")

def split_into_sentences(text: str) -> List[str]:
    """
    将长文本分割成句子
    """
    pattern = r'(?<=[。！？.!?])\s*'
    sentences = re.split(pattern, text)
    return [sent.strip() for sent in sentences if sent.strip()]

def get_sentence_embeddings(sentences: List[str], model, tokenizer, device, batch_size: int = 32) -> torch.Tensor:
    """
    批量计算句子的嵌入向量
    """
    embeddings = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            batch_embeddings = outputs.last_hidden_state[:, 0]
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings)
    
    return torch.cat(embeddings, dim=0)



def find_most_similar_sentences(
    target_sentence: str,
    paragraph: str,
    model,
    tokenizer,
    device,
    top_k: int = 3,
    batch_size: int = 32,
    min_length: int = 5  # 最小句子长度
) -> List[Tuple[str, float]]:
    """
    使用FAISS-GPU找出段落中与目标句子最相似的top_k个句子
    
    参数:
        target_sentence: 目标句子
        paragraph: 要分析的长段落
        model: 预训练模型
        tokenizer: 分词器
        device: 计算设备
        top_k: 返回相似度最高的句子数量
        batch_size: 批处理大小
        min_length: 最小句子长度
    
    返回:
        包含(句子, 相似度分数)元组的列表，按相似度降序排列
    """
    # 分句并过滤过短的句子
    sentences = [s for s in split_into_sentences(paragraph) if len(s) >= min_length]
    
    if not sentences:
        return []
    
    # 获取目标句子的嵌入向量
    target_embedding = get_sentence_embeddings([target_sentence], model, tokenizer, device)[0].unsqueeze(0)
    
    # 获取所有句子的嵌入向量
    sentence_embeddings = get_sentence_embeddings(sentences, model, tokenizer, device, batch_size)
    # pdb.set_trace()
    repeated_idx, repeated_val = find_same_features(target_embedding.detach().cpu().numpy(), sentence_embeddings.detach().cpu().numpy())
    
    # 组织返回结果
    results = []
    for sim, idx in zip(repeated_val[0], repeated_idx[0]):
        if idx >= 0:  # FAISS可能返回-1表示未找到足够的结果
            results.append((sentences[idx], float(sim)))
    
    return results

# 使用示例
if __name__ == "__main__":
    # 示例文本
    paragraph = """
    人工智能技术正在快速发展。它已经在多个领域展现出强大的潜力。
    从医疗诊断到自动驾驶，AI的应用无处不在。但我们也要警惕AI带来的风险。
    如何确保AI的发展方向符合人类利益，是一个重要课题。我们需要建立完善的AI伦理准则。
    在未来，AI将继续改变我们的生活方式。科技发展永远要以服务人类为本。
    """
    
    target = "AI技术在医疗领域有重要应用"
    
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