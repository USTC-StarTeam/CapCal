"""LLM 工具函数和 bias 实验入口集合。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 添加项目根目录到 sys.path，确保可以导入 src 模块
_current_file = Path(__file__).resolve()
_code_dir = _current_file.parent.parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# vLLM 导入（可选）
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

import json
import math,re
from typing import Optional, List, Dict
import yaml,csv
from transformers import DynamicCache
import copy

import numpy as np
from src.logger import get_logger
from src.util import (
    get_auto_device, RankLLM_format,should_generate_tokens_count,
    convert_to_int_indices,build_metric_comparison_report
    )
from src.BaseConfig import ConfigBase

class LLMExperiment:
    def __init__(self, args:ConfigBase):
        self.args = args #参数
        self.logger = get_logger(__name__, log_config=getattr(args, "log", None)) #日志文件

    def run(self):
        self.logger.info("LLMExperiment running...")
        self.logger.info(f"Args: {self.args}")
        self.logger.info(f"Logger: {self.logger}")

    def load_model(self) -> tuple[Optional[torch.nn.Module], Optional[AutoTokenizer], Optional[torch.device]]:
        """加载模型并返回 (model, tokenizer, device)。"""
        self.logger.info(f"🤖 加载 {self.args.model.model_name} 模型...")

        try:
            # 检查是否使用 vLLM
            use_vllm = getattr(self.args.model, 'use_vllm', False)
            if use_vllm and VLLM_AVAILABLE:
                tensor_parallel_size = getattr(self.args.model, 'tensor_parallel_size', 1)
                self.logger.info(f"🚀 使用 vLLM 加载模型，tensor_parallel_size={tensor_parallel_size}")
                
                # 验证 GPU 数量
                available_gpus = torch.cuda.device_count()
                self.logger.info(f"📊 系统检测到 {available_gpus} 个 GPU")
                if tensor_parallel_size > available_gpus:
                    self.logger.warning(
                        f"⚠️ 请求的 tensor_parallel_size={tensor_parallel_size} 超过可用 GPU 数量 {available_gpus}，"
                        f"将使用 {available_gpus} 个 GPU"
                    )
                    tensor_parallel_size = available_gpus
                
                self.model = LLM(
                    model=self.args.model.model_name,
                    tensor_parallel_size=tensor_parallel_size,
                    max_model_len=self.args.model.max_length,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.95,  # 使用 90% 的 GPU 内存
                    enable_prefix_caching=True,  # 启用前缀缓存以加速
                )
                self.tokenizer = self.model.get_tokenizer()
                self.use_vllm = True
                self.resolved_device = torch.device("cuda")
                self.logger.info(f"✅ vLLM 模型加载成功，使用 {tensor_parallel_size} 个 GPU 进行 tensor parallel 推理")
                return self.model, self.tokenizer, self.resolved_device
            elif use_vllm and not VLLM_AVAILABLE:
                self.logger.warning("⚠️ 请求使用 vLLM 但未安装，回退到标准 PyTorch 加载")
            
            # 标准 PyTorch 加载流程
            self.use_vllm = False
            self.resolved_device = get_auto_device(self.args.model.device)
            self.logger.info(f"使用设备: {self.resolved_device}")

            local_only = getattr(self.args.model, "local_files_only", True)
            self.logger.info(f"local_files_only={local_only}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model.model_name,
                use_fast=True,
                padding_side="left",
                model_max_length=self.args.model.max_length,
                local_files_only=local_only,
            )
            
            # 设置 pad_token（如果 tokenizer 没有 pad_token）
            pad_token_added = False
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.logger.info(f"⚠️  Tokenizer 没有 pad_token，使用 eos_token 作为 pad_token")
                else:
                    # 如果连 eos_token 都没有，添加一个新的 pad_token
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    pad_token_added = True
                    self.logger.info(f"⚠️  Tokenizer 没有 pad_token 和 eos_token，添加新的 pad_token '[PAD]'")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model.model_name,
                local_files_only=local_only,
            ).to(self.resolved_device)
            
            # 如果添加了新的 pad_token，需要调整模型的 embedding 层大小
            if pad_token_added:
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.logger.info(f"⚠️  已调整模型 embedding 层大小以适配新的 pad_token")

            if torch.cuda.device_count() > 1:
                self.logger.info(f"🚀 检测到 {torch.cuda.device_count()} 个 GPU，启用多 GPU 并行")
                self.model = torch.nn.DataParallel(self.model).module

            self.model.eval()
            self.logger.info(f"✅ 模型加载成功，使用设备: {self.resolved_device}")
            return self.model, self.tokenizer, self.resolved_device

        except Exception as exc:  # pragma: no cover - 仅用于捕获并记录异常
            self.logger.error(f"❌ 模型加载失败: {exc}", exc_info=True)
            import traceback

            traceback.print_exc()
            return None, None, None

    def get_passage_generation(self,query: str,passages: List[str]) -> str:
        """生成模型对 passages 的自然语言输出。"""
        rankllm_prompt = RankLLM_format(passages, query, self.args.model.prompt_format)
        self.logger.info(f"  🤖 模型提示词: {rankllm_prompt}")

        # 如果使用 vLLM
        if hasattr(self, 'use_vllm') and self.use_vllm:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=200,
                stop_token_ids=[self.tokenizer.eos_token_id] if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id else None,
            )
            outputs = self.model.generate([rankllm_prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            return generated_text

        # 标准 PyTorch 推理流程
        if self.args.model.model_name != "Qwen/Qwen3-8B":
            inputs = self.tokenizer(rankllm_prompt,return_tensors="pt",padding=True,truncation=True,max_length=self.args.model.max_length)
            inputs = {k: v.to(self.resolved_device) for k, v in inputs.items()}
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
                },
                {"role": "user", "content": rankllm_prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.resolved_device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            use_cache=self.args.model.cache_use,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = self.tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # 打印生成的 token id 序列，便于调试
        # print(f"generated_ids={generated_ids[0][inputs['input_ids'].shape[1]:]}")
        return generated_text

    def get_passage_logits_apply_bias_correction(self,query: str,passages: List[str],bias_rate: float = 1.0) -> List[int]:
        """在 logits 上应用 bias 校正并返回排序索引。"""
        same_passages = [self.args.experiment.same_passage_text] * len(passages)
        same_passages_prompt = RankLLM_format(same_passages, query, self.args.model.prompt_format)
        same_passages_inputs = self.tokenizer(
            same_passages_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.model.max_length,
        )
        same_passages_inputs = {k: v.to(self.resolved_device) for k, v in same_passages_inputs.items()}

        real_passages_prompt = RankLLM_format(passages, query, self.args.model.prompt_format)
        real_passages_inputs = self.tokenizer(
            real_passages_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.model.max_length,
        )
        real_passages_inputs = {k: v.to(self.resolved_device) for k, v in real_passages_inputs.items()}

        should_generate_tokens = should_generate_tokens_count(len(passages))#计算生成token数目
        decoded_text = ""
        with torch.no_grad():
            same_outputs = self.model(**same_passages_inputs, use_cache=self.args.model.cache_use)
            same_logits = same_outputs.logits[:, -1, :]
            same_past = same_outputs.past_key_values if self.args.model.cache_use else None

            real_outputs = self.model(**real_passages_inputs, use_cache=self.args.model.cache_use)
            real_logits = real_outputs.logits[:, -1, :]
            real_past = real_outputs.past_key_values if self.args.model.cache_use else None

            final_logits = real_logits

            generated_tokens= []
            next_token_id = torch.argmax(final_logits, dim=-1)

            same_base_input_ids = (
                same_passages_inputs["input_ids"] if not self.args.model.cache_use else None
            )
            real_base_input_ids = (
                real_passages_inputs["input_ids"] if not self.args.model.cache_use else None
            )

            for _ in range(should_generate_tokens):
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
                generated_tokens.append(next_token_id.item())

                if self.args.model.cache_use:
                    incr_input_ids=next_token_id.unsqueeze(0).to(self.resolved_device)
                    real_outputs = self.model(
                        input_ids=incr_input_ids,
                        use_cache=True,
                        past_key_values=real_past,
                    )
                    real_past = real_outputs.past_key_values
                    real_logits = real_outputs.logits[:, -1, :]

                    same_outputs = self.model(
                        input_ids=incr_input_ids,
                        use_cache=True,
                        past_key_values=same_past,
                    )
                    same_past = same_outputs.past_key_values
                    same_logits = same_outputs.logits[:, -1, :]
                else:
                    generated_tokens_tensor = torch.tensor([generated_tokens], device=self.resolved_device)
                    real_full_input_ids = torch.cat(
                        [real_base_input_ids, generated_tokens_tensor], dim=1
                    )
                    same_full_input_ids = torch.cat(
                        [same_base_input_ids, generated_tokens_tensor], dim=1
                    )

                    if real_full_input_ids.shape[1] > self.args.model.max_length:#如果生成的token数目大于模型最大长度，则截取模型最大长度
                        real_full_input_ids = real_full_input_ids[:, -self.args.model.max_length:]
                    if same_full_input_ids.shape[1] > self.args.model.max_length:#如果生成的token数目大于模型最大长度，则截取模型最大长度
                        same_full_input_ids = same_full_input_ids[:, -self.args.model.max_length:]

                    real_outputs = self.model(input_ids=real_full_input_ids)
                    real_logits = real_outputs.logits[:, -1, :]

                    same_outputs = self.model(input_ids=same_full_input_ids)
                    same_logits = same_outputs.logits[:, -1, :]

                final_logits = real_logits - same_logits * bias_rate
                next_token_id = torch.argmax(final_logits, dim=-1)

            if generated_tokens:
                decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                self.logger.info(f"  📝 bias 处理的文本: {decoded_text}")
            else:
                self.logger.error("  ⚠️  未生成任何 token（可能立即遇到 EOS 或没有生成）")
                return []

        keyword_pattern = None
        sorted_indices = convert_to_int_indices(decoded_text, keyword_pattern,len(passages))
        return sorted_indices

    def process_single_query(self,query, ir_metrics, **kwargs) -> Dict:
        """处理单个查询"""
        self.logger.info(f"🔍 处理查询: {query.query_id}")
    

        #查询获取
        passages = query.passages[: self.args.experiment.passages_per_query]
        relevance_scores = query.relevance_scores[: self.args.experiment.passages_per_query]

        self.logger.info(f"  📊 查询文本: {query.query_text[:1000]}...")
        self.logger.info(f"  📄 Passages数量: {len(passages)}")

        try:
            #正常生成
            generated_text = self.get_passage_generation(
                query.query_text, passages
            )
            real_sorted_indices = convert_to_int_indices(
                generated_text, None, len(passages)
            )
            if self.args.verbose:
                self.logger.info(f"  🤖 模型生成的回复:{generated_text}")

            #bias校正生成
            corrected_sorted_indices = self.get_passage_logits_apply_bias_correction(
                query.query_text,passages,
                self.args.experiment.bias_rate
            )  # 应用bias校正和排序结果

            # 评估性能
            self.logger.info(
                f"提取结果:{real_sorted_indices}, {corrected_sorted_indices}, {relevance_scores}"
            )
            comparison_result = ir_metrics.compare_orders(
                real_sorted_indices, corrected_sorted_indices, relevance_scores
            )

            # 提取性能指标
            original_performance = comparison_result["original"]
            corrected_performance = comparison_result["corrected"]
            performance_improvement = comparison_result["improvements"]

            result = {
                "query_id": query.query_id,
                "query_text": query.query_text,
                "passages": passages,
                "original_performance": original_performance,
                "corrected_performance": corrected_performance,
                "performance_improvement": performance_improvement,
            }

            # 生成指标对比字符串
            metric_report_str = build_metric_comparison_report(
                original_performance,
                corrected_performance,
                performance_improvement,
                ir_metrics.default_measures,
            )

            lines = [
                "✅ 处理完成",
                "=" * 60,
                "📊 评估指标对比表:",
                "=" * 60,
                f"{'指标名称':<10} {'原始值':<10} {'校正值':<10} {'提升值':<10}",
                "-" * 60,
                metric_report_str,
                "=" * 60,
            ]
            handler_info = "\n".join(lines)

            self.logger.info(handler_info)

            # 显示统计显著性信息
            if (
                "statistical_significance" in performance_improvement
                and performance_improvement["statistical_significance"]
            ):
                stats = performance_improvement["statistical_significance"]
                if "p_value" in stats:
                    significance_symbol = "***" if stats.get("significant", False) else ""
                    self.logger.info(
                        f"    📊 统计显著性: p={stats['p_value']:.4f} {significance_symbol}"
                    )
                    if "effect_size" in stats:
                        self.logger.info(f"    📊 效应大小: {stats['effect_size']:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"  ❌ 处理查询 {query.query_id} 时出错: {e}", exc_info=True)
            import traceback

            traceback.print_exc()
            return None

    def save_results(self,results: List[Dict], output_dir: str, ir_metrics,**kwargs):
        """保存结果"""
        self.logger.info(f"💾 保存结果到 {output_dir}")
        os.makedirs(output_dir, exist_ok=True)# 确保输出目录存在

        # 配置文件
        output_config = os.path.join(output_dir, "output_config.yaml")
        with open(output_config, "w", encoding="utf-8") as f:
            yaml.dump(self.args, f, indent=2)
        
        # 保存详细结果（CSV）：按 query 展示原始 / 校正 / 提升的指标
        output_file_name = kwargs.get("output_file_name", "bias_results.csv")
        results_file = os.path.join(output_dir, output_file_name)
        if results:
            # 表头：query_id, query_text, 各指标的 original / corrected / improvement
            fieldnames = ["query_id", "query_text"]
            metric_names = [str(m) for m in ir_metrics.default_measures]
            for name in metric_names:
                fieldnames.extend([
                    f"{name}_orig",
                    f"{name}_corr",
                    f"{name}_improv",
                ])

            with open(results_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for r in results:
                    if r is None:
                        continue
                    row = {
                        "query_id": r.get("query_id", ""),
                        "query_text": r.get("query_text", ""),
                    }
                    orig = r.get("original_performance", {}) or {}
                    corr = r.get("corrected_performance", {}) or {}
                    impr = r.get("performance_improvement", {}) or {}

                    for name in metric_names:
                        row[f"{name}_orig"] = orig.get(name, "")
                        row[f"{name}_corr"] = corr.get(name, "")
                        row[f"{name}_improv"] = impr.get(f"{name}_improvement", "")

                    writer.writerow(row)

        # 生成专业评估报告
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            self.logger.error("❌ 没有有效结果可保存")
            exit(0)
        # 使用ir_metrics生成专业报告
        evaluation_report = ir_metrics.generate_evaluation_report(valid_results)
        
        # 保存评估报告
        report_file = os.path.join(output_dir, "evaluation_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
        self.logger.info(f"✅ 结果已保存")

        # 显示汇总指标表格
        if "metrics_summary" in evaluation_report:
            metrics_imp = evaluation_report["metrics_summary"]

            # 1) 原始 / 校正绝对值均值与方差（直接从 valid_results 重新计算）
            # 按指标收集所有 query 的原始/校正值
            orig_values_map: Dict[str, List[float]] = {}
            corr_values_map: Dict[str, List[float]] = {}
            for r in valid_results:
                orig_perf = r.get("original_performance", {}) or {}
                corr_perf = r.get("corrected_performance", {}) or {}
                for measure in ir_metrics.default_measures:
                    name = str(measure)
                    if name in orig_perf and name in corr_perf:
                        orig_values_map.setdefault(name, []).append(orig_perf[name])
                        corr_values_map.setdefault(name, []).append(corr_perf[name])
            print("orig_values_map:",len(orig_values_map))
            print("corr_values_map:",len(corr_values_map))
            abs_lines = []
            abs_data = []  # 用于保存CSV数据
            for measure in ir_metrics.default_measures:
                measure_name = str(measure)
                orig_vals = orig_values_map.get(measure_name, [])
                corr_vals = corr_values_map.get(measure_name, [])
                if not orig_vals or not corr_vals:
                    continue

                orig_mean = np.mean(orig_vals)
                corr_mean = np.mean(corr_vals)

                # 使用无偏样本标准差（n>1 时），n<=1 时记为 0
                # 使用 numpy 计算更为简洁和健壮
                orig_std = float(np.std(orig_vals, ddof=1)) if len(orig_vals) > 1 else 0.0
                corr_std = float(np.std(corr_vals, ddof=1)) if len(corr_vals) > 1 else 0.0

                abs_line = (
                    f"  {measure_name:<15} "
                    f"{orig_mean:<12.4f} {orig_std:<12.4f} "
                    f"{corr_mean:<12.4f} {corr_std:<12.4f}"
                )
                abs_lines.append(abs_line)
                
                # 保存数据用于CSV
                abs_data.append({
                    "指标名称": measure_name,
                    "原始均值": f"{orig_mean:.4f}",
                    "原始Std": f"{orig_std:.4f}",
                    "校正均值": f"{corr_mean:.4f}",
                    "校正Std": f"{corr_std:.4f}"
                })

            if abs_lines:
                abs_table = [
                    "📊 绝对性能均值与方差汇总表:",
                    "=" * 80,
                    f"{'指标名称':<15} {'原始均值':<12} {'原始Std':<12} {'校正均值':<12} {'校正Std':<12}",
                    "-" * 80,
                    "\n".join(abs_lines),
                    "=" * 80,
                ]
                self.logger.info("\n".join(abs_table))
                
                # 保存绝对性能汇总表到CSV文件
                abs_summary_file = os.path.join(output_dir, "absolute_performance_summary.csv")
                with open(abs_summary_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["指标名称", "原始均值", "原始Std", "校正均值", "校正Std"])
                    writer.writeheader()
                    writer.writerows(abs_data)
                self.logger.info(f"✅ 绝对性能汇总表已保存到 {abs_summary_file}")
                

            # 2) 提升情况统计（保留原有表）
            imp_lines = []
            for measure in ir_metrics.default_measures:
                measure_name = str(measure)
                improvement_key = f"{measure_name}_improvement"

                if improvement_key in metrics_imp:
                    stats = metrics_imp[improvement_key]
                    positive_rate = (stats["positive_improvements"] / len(valid_results)) * 100

                    imp_line = (
                        f"  {measure_name:<20} {stats['mean']:<12.4f} {stats['std']:<12.4f} "
                        f"{stats['positive_improvements']:<10} {stats['negative_improvements']:<10} {positive_rate:<12.2f}"
                    )
                    imp_lines.append(imp_line)

            if imp_lines:
                imp_table = [
                    "📊 整体性能提升汇总表:",
                    "=" * 80,
                    f"{'指标名称':<15} {'平均提升':<12} {'标准差':<12} {'正提升数':<10} {'负提升数':<10} {'正提升率(%)':<12}",
                    "-" * 80,
                    "\n".join(imp_lines),
                    "=" * 80,
                ]
                self.logger.info("\n".join(imp_table))

            self.logger.info(f"  📈 总查询数: {len(valid_results)}")
            
class LLMExp_StateMachine(LLMExperiment):
    def __init__(self, args: ConfigBase):
        super().__init__(args)
        # 允许 list，存储 [id_1, id_0]
        self.candidate_tokens_map: Dict[int, List[int]] = {}   # 数字的后缀 tokens（去掉前缀）
        self._bracket_token_ids: Dict[str, int] = {}
        self.l_bracket_id = None
        self.l_bracket_id_with_special_tokens = None
        self.r_bracket_id = None
        self.gt_id = None

    def load_model(self):
        # 检查是否使用 vLLM，但需要访问 logits 的方法不支持 vLLM
        use_vllm = getattr(self.args.model, 'use_vllm', False)
        if use_vllm:
            self.logger.warning(
                "⚠️ 检测到使用 vLLM，但 LLMExp_StateMachine 需要访问模型 logits，"
                "vLLM 不支持直接访问 logits。将回退到标准 PyTorch 加载。"
            )
            # 强制禁用 vLLM
            self.args.model.use_vllm = False
        
        result = super().load_model()
        if self.tokenizer:
            self.pre_cache_token_ids()
        return result

    def pre_cache_token_ids(self):
        """
        预缓存 Token ID (保留之前的上下文修复逻辑)
        """
        if self.tokenizer is None: return

        # 1. 缓存特殊符号
        def get_id(s):
            ids = self.tokenizer.get_vocab().get(s)
            return ids if ids else None

        def get_id_with_special_tokens(s):
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            return ids[-1] if ids else None

        self.gt_id = get_id_with_special_tokens(">")     
        self.l_bracket_id_with_special_tokens = get_id_with_special_tokens("[")
        self.l_bracket_id = get_id("[") 
        self.r_bracket_id = get_id("]")
        self.logger.info(f"🧩 Special Tokens: [={self.l_bracket_id}, ]={self.r_bracket_id}, >={self.gt_id}")

        # 2. 缓存数字 token
        max_passages = getattr(self.args.experiment, "passages_per_query", 50)
        self.candidate_tokens_map = {}
        
        for i in range(0, max_passages + 1):
            try:
                num_ids = self.tokenizer.encode(str(i), add_special_tokens=False)

                if not num_ids:
                    continue

                self.candidate_tokens_map[i] = num_ids
            except Exception as e:
                self.logger.error(f"Error caching {i}: {e}")
        self.logger.info(f"🧩 Token Map Cached (1-{max_passages})\nself.candidate_tokens_map={self.candidate_tokens_map}")
    
    def clone_dynamic_cache(self, cache: DynamicCache) -> DynamicCache:
        """
        创建一个 DynamicCache 的深拷贝。
        使用 copy.deepcopy 作为最安全的方法。
        """
        return copy.deepcopy(cache)
    
    def calculate_sequence_prob(self, past_key_values, start_token_id: int) -> float:
        """
        直接在概率空间计算序列的条件概率乘积。
        返回: Π p(token_i | context)
        """
        prob_cache = {}
        curr_past = self.clone_dynamic_cache(past_key_values)
        curr_input_id = start_token_id
        input_tensor = torch.tensor([[curr_input_id]], device=self.resolved_device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor, use_cache=True, past_key_values=curr_past)
            logits = outputs.logits[:, -1, :]
            curr_past = outputs.past_key_values
            probs = F.softmax(logits, dim=-1)
            # 获取数字token1-9的生成概率
            digit_tokens = [self.candidate_tokens_map[i][0] for i in range(0, 10)]
            # print(f"digit_tokens={digit_tokens}")
            digit_probs = probs[0, digit_tokens]
            for tokenid, prob in zip(digit_tokens, digit_probs):# 打印不同token的概率
                prob_cache[f"{self.tokenizer.decode([tokenid], skip_special_tokens=False)}|context"] = prob.item()
            
            # 额外计算指定数字序列（含右括号）的概率：1],2],3],9],11-19],20]
            complex_mask = digit_tokens +[self.r_bracket_id]
            for num in range(1,10):
                new_past = self.clone_dynamic_cache(curr_past)
                # print(f"---------------num={num}---------------------")
                # 每次循环都从 curr_past 开始（DynamicCache 不需要 copy，每次前向传播会返回新对象）
                next_token_id = self.candidate_tokens_map[num][0]
                incr_input_ids = torch.tensor([[next_token_id]], device=self.resolved_device)
                outputs = self.model(
                    input_ids=incr_input_ids, 
                    use_cache=True, 
                    past_key_values=new_past 
                )
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                if num == 1:
                    complex_probs = probs[0, complex_mask]
                    for tokenid, prob in zip(complex_mask, complex_probs):
                        prob_cache[f"{self.tokenizer.decode([tokenid], skip_special_tokens=False)}|{num},context"] = prob.item()
                elif num == 2:
                    _mask=self.candidate_tokens_map[0]+[self.r_bracket_id]
                    _probs = probs[0, _mask]
                    for tokenid, prob in zip(_mask, _probs):
                        prob_cache[f"{self.tokenizer.decode([tokenid], skip_special_tokens=False)}|{num},context"] = prob.item()
                else:
                    _probs = probs[0, [self.r_bracket_id]]
                    for tokenid, prob in zip([self.r_bracket_id], _probs):
                        prob_cache[f"{self.tokenizer.decode([tokenid], skip_special_tokens=False)}|{num},context"] = prob.item()
        return prob_cache

    def evaluate_candidates(self, 
                            candidates: List[int], 
                            bias_rate: float,
                            real_past, same_past, 
                            last_token_id: int) -> int:
        """
        在 P (概率) 层面进行优化
        评估序列包括数字 tokens + r_bracket_id
        """
        best_num = -1

        # 计算并缓存条件概率
        real_prob_cache = self.calculate_sequence_prob(real_past, last_token_id)
        same_prob_cache = self.calculate_sequence_prob(same_past, last_token_id)
        real_prob={}
        same_prob={}
        final_prob={}

        def joint_prob(num: int, prob_cache: dict) -> float:
            if num<10:
                p = prob_cache.get(f"{num}|context", 0.0)*prob_cache.get(f"]|{num},context", 0.0)
                return p
            else:
                strnum = str(num)
                p = prob_cache.get(f"{strnum[0]}|context", 0.0)*prob_cache.get(f"{strnum[1]}|{strnum[0]},context", 0.0)
                return p
        # self.logger.info('--------------------------------')
        for num in candidates:
            real_p = joint_prob(num, real_prob_cache)
            real_prob[num] = real_p
            same_p = joint_prob(num, same_prob_cache)
            same_prob[num] = same_p
            final_prob[num] = real_p - bias_rate*(same_p - 1/len(candidates))
            # self.logger.info(f"{num}:{real_p:.4f}|{same_p:.4f}|{final_prob[num]:.4f}|")
        # print(f"real_prob={real_prob}")
        # print(f"final_prob={final_prob}")
        # input()
        best_num = max(final_prob, key=final_prob.get)
        return best_num

    def _feed_tokens(self, token_ids: List[int], real_past, same_past=None):
        """推进 KV Cache (处理 list)"""
        if not token_ids: return real_past, same_past
        
        flat_ids = []
        for x in token_ids:
            if isinstance(x, list): flat_ids.extend(x)
            else: flat_ids.append(x)
            
        inp = torch.tensor([flat_ids], device=self.resolved_device)
        with torch.no_grad():
            r_out = self.model(input_ids=inp, use_cache=True, past_key_values=real_past)
            new_r_past = r_out.past_key_values
            
            new_s_past = None
            if same_past is not None:
                s_out = self.model(input_ids=inp, use_cache=True, past_key_values=same_past)
                new_s_past = s_out.past_key_values
        return new_r_past, new_s_past
    
    def get_passage_generation(self, query: str, passages: List[str]) -> List[int]:
        def joint_prob(num: int, prob_cache: dict) -> float:
            if num<10:
                p = prob_cache.get(f"{num}|context", 0.0)*prob_cache.get(f"]|{num},context", 0.0)
                return p
            else:
                strnum = str(num)
                p = prob_cache.get(f"{strnum[0]}|context", 0.0)*prob_cache.get(f"{strnum[1]}|{strnum[0]},context", 0.0)
                return p
        """
        获取 passages 的 generation
        """
        # 1. 构造 prompt 与输入（只走 real 路径，不做 bias 校正）
        prompt_real = RankLLM_format(passages, query, self.args.model.prompt_format)
        inputs_real = self.tokenizer(
            prompt_real,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.model.max_length,
        ).to(self.resolved_device)

        # 2. Prefill
        with torch.no_grad():
            real_out = self.model(**inputs_real, use_cache=True)
        real_past = real_out.past_key_values

        # 3. 循环（约束解码，格式 [i]>[j]>...）
        candidates = list(range(1, len(passages) + 1))
        decoded_indices: List[int] = []
        all_generated_tokens: List[int] = []

        for step in range(len(passages)):
            # Step A: 追加 ">" （从第二个元素开始）
            if step > 0:
                all_generated_tokens.append(self.gt_id)
                real_past, _ = self._feed_tokens([self.gt_id], real_past, None)

            # Step B: 追加 "["（首个使用 vocab 形式，后续使用 tokenizer 编码形式）
            if step == 0:
                all_generated_tokens.append(self.l_bracket_id)
                real_past, _ = self._feed_tokens([self.l_bracket_id], real_past, None)
            else:
                all_generated_tokens.append(self.l_bracket_id_with_special_tokens)
                real_past, _ = self._feed_tokens([self.l_bracket_id_with_special_tokens], real_past, None)

            # Step C: 选择候选编号
            prob = self.calculate_sequence_prob(real_past, self.l_bracket_id)
            candidates_prob = {num: joint_prob(num, prob) for num in candidates}
            # print(f"candidates_prob={candidates_prob}")
            best_num = max(candidates_prob, key=candidates_prob.get)
            decoded_indices.append(best_num)

            # Step D: 追加数字与右括号
            best_tokens = self.candidate_tokens_map[best_num] + [self.r_bracket_id]
            all_generated_tokens.extend(best_tokens)
            real_past, _ = self._feed_tokens(best_tokens, real_past, None)

            if best_num in candidates:
                candidates.remove(best_num)
            if not candidates:
                break

        # 4. 日志记录
        if all_generated_tokens:
            decoded_text = self.tokenizer.decode(all_generated_tokens, skip_special_tokens=True)
            self.logger.info(f" 📝 Constrained Gen: {decoded_text}")
        else:
            self.logger.error(" ⚠️ 未生成任何内容")

        return decoded_indices

    def get_passage_logits_apply_bias_correction(self, query: str, passages: List[str], bias_rate: float = 1.0) -> List[int]:
        # 1. 准备 Prompt & Inputs
        same_passages = [self.args.experiment.same_passage_text] * len(passages)

        prompt_same = RankLLM_format(same_passages, query, self.args.model.prompt_format)
        prompt_real = RankLLM_format(passages, query, self.args.model.prompt_format)
        
        self.logger.info(f"prompt_same={prompt_same}")
        self.logger.info(f"prompt_real={prompt_real}")
        
        inputs_same = self.tokenizer(prompt_same, return_tensors="pt", padding=True, truncation=True, max_length=self.args.model.max_length).to(self.resolved_device)
        inputs_real = self.tokenizer(prompt_real, return_tensors="pt", padding=True, truncation=True, max_length=self.args.model.max_length).to(self.resolved_device)

        # 2. Prefill
        with torch.no_grad():
            real_out = self.model(**inputs_real, use_cache=True)
            same_out = self.model(**inputs_same, use_cache=True)
        
        real_past = real_out.past_key_values
        same_past = same_out.past_key_values

        # 3. 循环
        candidates = list(range(1, len(passages) + 1))
        decoded_indices = []
        all_generated_tokens = []
        
        for step in range(len(passages)):
            # Step A: Feed " >"
            if step > 0:
                all_generated_tokens.append(self.gt_id)
                real_past, same_past = self._feed_tokens([self.gt_id], real_past, same_past)
            
            # Step B: Feed " ["
            if step == 0:
                all_generated_tokens.append(self.l_bracket_id)
                real_past, same_past = self._feed_tokens([self.l_bracket_id], real_past, same_past)
            else:
                all_generated_tokens.append(self.l_bracket_id_with_special_tokens)
                real_past, same_past = self._feed_tokens([self.l_bracket_id_with_special_tokens], real_past, same_past)
            
            # Step C: Evaluate (With Geometric Mean Normalization)
            # print("--------------------------------")
            # print(f"step={step}, candidates={candidates}")
            best_num = self.evaluate_candidates(
                candidates, bias_rate, 
                real_past, same_past, 
                last_token_id=self.l_bracket_id
            )
            decoded_indices.append(best_num)
            
            # Step D: Feed Num + "]"
            best_tokens = self.candidate_tokens_map[best_num]+[self.r_bracket_id]
            all_generated_tokens.extend(best_tokens)

            tokens_to_feed = best_tokens
            real_past, same_past = self._feed_tokens(tokens_to_feed, real_past, same_past)
            
            if best_num in candidates:
                candidates.remove(best_num)
            if not candidates:
                break
                
        # Log
        if all_generated_tokens:
            text = self.tokenizer.decode(all_generated_tokens, skip_special_tokens=True)
            self.logger.info(f" 📝 Bias Generation: {text}")

        return decoded_indices

    def process_single_query(self,query, ir_metrics, **kwargs) -> Dict:
        """处理单个查询"""
        self.logger.info(f"🔍 处理查询: {query.query_id}")

        #查询获取
        passages = query.passages[: self.args.experiment.passages_per_query]
        relevance_scores = query.relevance_scores[: self.args.experiment.passages_per_query]

        self.logger.info(f"  📊 查询文本: {query.query_text[:1000]}...")
        self.logger.info(f"  📄 Passages数量: {len(passages)}")

        try:
            real_sorted_indices = self.get_passage_generation(
                query.query_text, passages
            )

            corrected_sorted_indices = self.get_passage_logits_apply_bias_correction(
                query.query_text,passages,
                self.args.experiment.bias_rate
            )

            self.logger.info(
                f"提取结果: {real_sorted_indices}, {corrected_sorted_indices}, {relevance_scores}"
            )
            comparison_result = ir_metrics.compare_orders(
                real_sorted_indices, corrected_sorted_indices, relevance_scores
            )

            # 提取性能指标
            original_performance = comparison_result["original"]
            corrected_performance = comparison_result["corrected"]
            performance_improvement = comparison_result["improvements"]

            result = {
                "query_id": query.query_id,
                "query_text": query.query_text,
                "passages": passages,
                "original_performance": original_performance,
                "corrected_performance": corrected_performance,
                "performance_improvement": performance_improvement,
            }

            # 生成指标对比字符串
            metric_report_str = build_metric_comparison_report(
                original_performance,
                corrected_performance,
                performance_improvement,
                ir_metrics.default_measures,
            )

            lines = [
                "✅ 处理完成",
                "=" * 60,
                "📊 评估指标对比表:",
                "=" * 60,
                f"{'指标名称':<10} {'原始值':<10} {'校正值':<10} {'提升值':<10}",
                "-" * 60,
                metric_report_str,
                "=" * 60,
            ]
            handler_info = "\n".join(lines)

            self.logger.info(handler_info)

            # 显示统计显著性信息
            if (
                "statistical_significance" in performance_improvement
                and performance_improvement["statistical_significance"]
            ):
                stats = performance_improvement["statistical_significance"]
                if "p_value" in stats:
                    significance_symbol = "***" if stats.get("significant", False) else ""
                    self.logger.info(
                        f"    📊 统计显著性: p={stats['p_value']:.4f} {significance_symbol}"
                    )
                    if "effect_size" in stats:
                        self.logger.info(f"    📊 效应大小: {stats['effect_size']:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"  ❌ 处理查询 {query.query_id} 时出错: {e}", exc_info=True)
            import traceback

            traceback.print_exc()
            return None

class LLMExp_DynamicBias(LLMExp_StateMachine):
    def __init__(self, args: ConfigBase):
        super().__init__(args)
        # 定义动态调整的参数
        # 基础 bias_rate 从配置读取，作为基准
        self.base_bias_rate = self.args.experiment.bias_rate
        
        # 敏感度系数：控制熵对 bias_rate 的影响程度
        # 如果 args 中没有配置，默认给一个值，例如 0.5
        self.sensitivity = getattr(self.args.experiment, "bias_sensitivity", 0.5)
        
        # 限制 bias_rate 的最大值，防止矫枉过正
        self.max_bias_rate = getattr(self.args.experiment, "max_bias_rate", 2.0)

    def calculate_entropy(self, probs: List[float]) -> float:
        """
        计算概率分布的熵 (Entropy) 作为不确定性的度量。
        Entropy = - sum(p * log(p))
        """
        # 1. 归一化概率（确保 sum=1）
        total_p = sum(probs)
        if total_p == 0:
            return 0.0
        
        normalized_probs = [p / total_p for p in probs]
        
        # 2. 计算熵
        entropy = 0.0
        for p in normalized_probs:
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    def get_dynamic_rate(self, entropy: float, num_candidates: int) -> float:
        """
        根据熵计算动态 bias rate。
        逻辑：
        1. 计算最大可能的熵 (均匀分布时的熵 = log(N))。
        2. 计算相对熵 (0~1 之间)。
        3. 相对熵越高，Bias Rate 越高。
        """
        if num_candidates <= 1:
            return self.base_bias_rate

        max_entropy = math.log(num_candidates)
        # 相对不确定性 (0.0 - 1.0)
        relative_uncertainty = entropy / max_entropy if max_entropy > 0 else 0
        
        # 动态调整公式：
        # Current_Rate = Base_Rate + (Sensitivity * Relative_Uncertainty)
        # 你也可以根据需求改为乘法： Base_Rate * (1 + Relative_Uncertainty)
        
        #dynamic_rate = self.base_bias_rate + (self.sensitivity * relative_uncertainty)
        dynamic_rate = min(5*relative_uncertainty, 2.0)
        # 截断范围
        return min(max(dynamic_rate, 0.0), self.max_bias_rate)

    def evaluate_candidates(self, 
                            candidates: List[int], 
                            bias_rate: float, # 这里传入的 bias_rate 会被我们动态计算的值覆盖
                            real_past, same_past, 
                            last_token_id: int) -> int:
        """
        重写评估候选者的方法，加入动态 Bias 逻辑。
        """
        best_num = -1

        # 1. 计算所有候选者的路径概率
        real_prob_cache = self.calculate_sequence_prob(real_past, last_token_id)
        same_prob_cache = self.calculate_sequence_prob(same_past, last_token_id)
        
        real_prob = {}
        same_prob = {}
        
        # 内部函数：计算联合概率
        def joint_prob(num: int, prob_cache: dict) -> float:
            if num < 10:
                p = prob_cache.get(f"{num}|context", 0.0) * prob_cache.get(f"]|{num},context", 0.0)
                return p
            else:
                strnum = str(num)
                # 处理两位数的情况
                p = prob_cache.get(f"{strnum[0]}|context", 0.0) * prob_cache.get(f"{strnum[1]}|{strnum[0]},context", 0.0)
                return p

        # 2. 收集所有候选者的概率
        candidate_real_probs = []
        for num in candidates:
            r_p = joint_prob(num, real_prob_cache)
            s_p = joint_prob(num, same_prob_cache)
            real_prob[num] = r_p
            same_prob[num] = s_p
            candidate_real_probs.append(r_p)

        # 3. ---> 核心修改：计算动态 Bias Rate <---
        current_entropy = self.calculate_entropy(candidate_real_probs)
        dynamic_bias_rate = self.get_dynamic_rate(current_entropy, len(candidates))
        
        # 记录日志，观察动态调整情况（可选，调试用）
        # self.logger.info(f"Step Candidates: {len(candidates)}, Entropy: {current_entropy:.4f}, Dynamic Rate: {dynamic_bias_rate:.4f}")

        # 4. 应用校正公式
        final_prob = {}
        baseline_prob = 1.0 / len(candidates) if candidates else 0.0
        
        for num in candidates:
            # 公式: P_final = P_real - Dynamic_Rate * (P_same - Baseline)
            final_prob[num] = real_prob[num] - dynamic_bias_rate * (same_prob[num] - baseline_prob)

        # 5. 选择最佳候选
        best_num = max(final_prob, key=final_prob.get)
        return best_num
            
LLMExp_FixedBias = LLMExp_StateMachine
LLMExp_AdaptiveBias = LLMExp_DynamicBias
