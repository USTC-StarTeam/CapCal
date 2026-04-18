import random
import numpy as np
import torch
from typing import List, Dict, Iterable, Optional
import re,datetime

INDEX_PATTERN= re.compile(r"\d+")# 查找passage标签序号，匹配[0-9]+的正则表达式

def set_seed(seed: int):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enable =True # 启用cuDNN加速
    torch.backends.cudnn.benchmark = True # 启用cuDNN自动优化

def get_auto_device(device: str = "auto") -> torch.device:
    """自动选择设备"""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)

def rerank_sorted(text,other_pattern = re.compile("<\|assistan\t\|>.-*?|<\|assistant\|>")):
    if other_pattern:
        text = re.sub(other_pattern,r"", text)
    return re.findall(INDEX_PATTERN, text)

def RankLLM_format(passages: List[str], query: str, prompt_format: str) -> str:
    # 使用RankLLM格式
    passages_text = ""
    for i, passage in enumerate(passages, start=1):
        passages_text += f"[{i}] {passage}\n"
    rankllm_prompt = prompt_format.format(passages_count=len(passages), query=query, passages_text=passages_text)
    return rankllm_prompt

def should_generate_tokens_count(passages_count: int) -> int:#计算生成token数目
    num_parentheses_and_greater_than = passages_count*3-1#括号数目+>数目
    # passage编号位数总和（从0到n-1）
    num_digits = sum(len(str(i)) for i in range(passages_count))
    return  num_parentheses_and_greater_than + num_digits

def convert_to_int_indices(text: str,keyword_pattern: re.Pattern,total_passages: int) -> List[int]:
    """提取排序结果并保证标签合法。

    - 丢弃所有大于 total_passages 的标签
    - 保留 1~total_passages 内出现过的标签（去重，保持顺序）
    - 将缺失的合法标签追加到序列末尾
    """
    raw_indices = list(map(int, rerank_sorted(text)))#提取排序结果并保证标签合法, keyword_pattern
    total_passages = max(int(total_passages or 0), 0)

    filtered: List[int] = []
    seen = set()
    for idx in raw_indices:
        if idx <= 0 or idx > total_passages:
            continue
        if idx in seen:
            continue
        filtered.append(idx)
        seen.add(idx)

    if total_passages > 0:
        missing = [i for i in range(1, total_passages + 1) if i not in seen]
        filtered.extend(missing)

    return filtered#返回合法的排序结果

def build_metric_comparison_report(
    original_performance: Dict,
    corrected_performance: Dict,
    performance_improvement: Dict,
    measures: Iterable) -> str:
    """Return formatted comparison lines for metrics.

    Builds lines like:
        <metric_name> <original> <corrected> <improvement>
    in a single string joined by newlines.
    """
    metric_lines: List[str] = []
    for measure in measures:
        measure_name = str(measure)
        if measure_name in original_performance:
            original_val = original_performance[measure_name]
            corrected_val = corrected_performance.get(measure_name, 0)
            improvement = performance_improvement.get(f"{measure_name}_improvement", 0)
            metric_lines.append(
                f"    {measure_name:<15} {original_val:<10.4f} {corrected_val:<10.4f} {improvement:<10.4f}"
            )
    return "\n".join(metric_lines)
    
def debug_logit_report(tokenizer, real_logits: torch.Tensor, final_logits: torch.Tensor,
                       generated_tokens: List[int], step: int, top_k: int = 5) -> None:
    """打印每一步 real 与 final logits 的 Top-K token 以及当前生成文本。"""
    if generated_tokens:
        decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        logger.info(f"  🤖 生成了 {len(generated_tokens)} 个token，解码后的文本: {decoded_text}")
    else:
        logger.info(f"  🤖 未生成任何token（可能立即遇到EOS）")

    # 处理 real_logits 的 batch 维度为 1D
    if len(real_logits.shape) == 2 and real_logits.shape[0] > 1:
        logits_1d_real = real_logits[0]
    else:
        logits_1d_real = real_logits.squeeze(0) if len(real_logits.shape) > 1 else real_logits

    top_k_real = min(top_k, logits_1d_real.shape[-1])
    top_logits_real, top_indices_real = torch.topk(logits_1d_real, k=top_k_real)
    probs_real = torch.nn.functional.softmax(logits_1d_real, dim=-1)
    top_probs_real = probs_real[top_indices_real]

    text_split= [f"  🔎 Step {step + 1} Real logits（未校正）- Top tokens:"]
    for i, (tid, lgt, pr) in enumerate(zip(top_indices_real, top_logits_real, top_probs_real)):
        ttext = tokenizer.decode([tid.item()], skip_special_tokens=False)
        text_split.append(f"    {i+1}. Token ID: {tid.item():6d} | Text: {repr(ttext):20s} | Logit: {lgt.item():8.4f} | Prob: {pr.item():.6f}")
    logger.info("\n".join(text_split))

    # 处理 final_logits 的 batch 维度为 1D
    if len(final_logits.shape) == 2 and final_logits.shape[0] > 1:
        logits_1d_final = final_logits[0]
    else:
        logits_1d_final = final_logits.squeeze(0) if len(final_logits.shape) > 1 else final_logits

    top_k_final = min(top_k, logits_1d_final.shape[-1])
    top_logits_final, top_indices_final = torch.topk(logits_1d_final, k=top_k_final)
    probs_final = torch.nn.functional.softmax(logits_1d_final, dim=-1)
    top_probs_final = probs_final[top_indices_final]

    text_split= [f"  🔎 Step {step + 1} Final logits（校正后）- Top tokens:"]
    for i, (tid, lgt, pr) in enumerate(zip(top_indices_final, top_logits_final, top_probs_final)):
        ttext = tokenizer.decode([tid.item()], skip_special_tokens=False)
        text_split.append(f"    {i+1}. Token ID: {tid.item():6d} | Text: {repr(ttext):20s} | Logit: {lgt.item():8.4f} | Prob: {pr.item():.6f}")
    logger.info("\n".join(text_split))
