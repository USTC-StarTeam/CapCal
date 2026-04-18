#!/usr/bin/env python3
"""
专业的IR评估指标包装器
专门使用ir-measures库进行标准化的信息检索评估
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import warnings
from collections import defaultdict
import ir_measures
from ir_measures import AP, NDCG, P, RR, R

class IRMetricsWrapper:
    """
    专业的IR评估指标包装器
    专门使用ir-measures库进行标准化评估
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        """
        初始化IR评估包装器
        
        Args:
            k_values: 评估的k值列表，默认为[1, 3, 5, 10]
        """
        self.k_values = k_values
        self.default_measures = self._get_default_measures()
        
        print("✅ 使用ir-measures进行专业IR评估")
        print(f"  - 评估k值: {k_values}")
        print(f"  - 评估指标: {[str(m) for m in self.default_measures]}")
    
    def _get_default_measures(self) -> List:
        """
        获取默认评估指标列表
        
        # 包含以下信息检索评估指标:
        #   AP: Average Precision（平均精度）
        #       - 衡量检索结果排序中相关文档出现的平均位置。越高表示相关文档靠前分布越集中。
        #   RR: Reciprocal Rank（倒数排名）
        #       - 只关注第一个相关文档出现的位置，数值为1/第一个相关文档排名。用于评估检索系统响应首个相关项目的能力。
        #   NDCG@k: Normalized Discounted Cumulative Gain at k（前k条的标准化折损累积增益）
        #       - 适用于排序型和非二值相关性。评估前k条排序结果的相关性，并做衰减和归一化。完美排序为1。
        #   P@k: Precision at k（前k个结果的精度）
        #       - 统计前k个返回结果中相关文档的比例。用于反映结果的准确性。
        """
        measures = [AP(), RR()] 
        
        # 添加NDCG@k指标
        for k in self.k_values:
            measures.append(NDCG@k)
        
        # 添加Precision@k指标
        for k in self.k_values:
            measures.append(P@k)
        
        return measures
    
    def _prepare_qrels(self, relevance_scores: List[int], query_id: str = "q1") -> Dict:
        """
        准备qrels格式数据
        
        Args:
            relevance_scores: 相关性分数列表
            query_id: 查询ID
            
        Returns:
            qrels格式的字典
        """
        qrels = {query_id: {}}
        for i, score in enumerate(relevance_scores):
            if score > 0:  # 只包含相关文档
                doc_id = f"d{i}"
                qrels[query_id][doc_id] = score
        return qrels
    
    def _prepare_run(self, logits: List[float], query_id: str = "q1") -> Dict:
        """
        准备run格式数据
        
        Args:
            logits: 模型输出的logits分数
            query_id: 查询ID
            
        Returns:
            run格式的字典
        """
        run = {query_id: {}}
        for i, score in enumerate(logits):
            doc_id = f"d{i}"
            run[query_id][doc_id] = float(score)
        return run
    
    def _normalize_sorted_indices(self, sorted_indices: List[int], num_docs: int) -> List[int]:
        """将可能的1-based索引转换为0-based，并过滤越界与重复。"""
        if not sorted_indices:
            return []

        # 判断是否为1-based：若不存在0且最大值==num_docs，则减1
        max_idx = max(sorted_indices)
        is_one_based = (0 not in sorted_indices) and (max_idx == num_docs)
        norm = [(i - 1) if is_one_based else i for i in sorted_indices]

        # 过滤越界，并去重保持顺序
        seen = set()
        result: List[int] = []
        for i in norm:
            if 0 <= i < num_docs and i not in seen:
                seen.add(i)
                result.append(i)
        return result

    def evaluate_ranking(self, logits: List[float], relevance_scores: List[int], 
                        k_values: Optional[List[int]] = None) -> Dict[str, float]:
        """
        评估排序性能
        
        Args:
            logits: 模型输出的logits分数
            relevance_scores: 相关性分数列表
            k_values: 评估的k值列表，如果为None则使用默认值
            
        Returns:
            评估指标字典
        """
        if k_values is not None:
            self.k_values = k_values
            self.default_measures = self._get_default_measures()
        
        # 准备数据
        qrels = self._prepare_qrels(relevance_scores)
        run = self._prepare_run(logits)

        # 计算指标
        results = ir_measures.calc_aggregate(self.default_measures, qrels, run)
        
        
        # 转换为字典格式
        metrics = {}
        for measure, value in results.items():
            metrics[str(measure)] = float(value)
        
        return metrics
    
    def _run_from_sorted_indices(self, sorted_indices: List[int], query_id: str = "q1") -> Dict:
        """
        根据已给定的排序索引列表构造 run（分数仅用于保序，前高后低）。
        """
        n = len(sorted_indices)
        scores = {idx: float(n - rank) for rank, idx in enumerate(sorted_indices)}
        run = {query_id: {f"d{idx}": score for idx, score in scores.items()}}
        return run

    def evaluate_order(self, sorted_indices: List[int], relevance_scores: List[int],
                       k_values: Optional[List[int]] = None) -> Dict[str, float]:
        """
        直接基于排序索引进行评估。
        sorted_indices: 模型输出的文档索引顺序（如 [3,1,2] 表示 d3 > d1 > d2）
        relevance_scores: 与原始文档索引对齐的相关性分数（index i 对应 di）
        """
        if k_values is not None:
            self.k_values = k_values
            self.default_measures = self._get_default_measures()

        sorted_indices = self._normalize_sorted_indices(sorted_indices, len(relevance_scores))
        qrels = self._prepare_qrels(relevance_scores)# 构造 qrels
        run = self._run_from_sorted_indices(sorted_indices)# 构造 run

        # 计算指标
        results = ir_measures.calc_aggregate(self.default_measures, qrels, run)

        # 转换为字典
        metrics = {str(measure): float(value) for measure, value in results.items()}
        return metrics

    def compare_orders(self, original_sorted_indices: List[int], corrected_sorted_indices: List[int],
                       relevance_scores: List[int], k_values: Optional[List[int]] = None) -> Dict[str, Dict[str, float]]:
        """
        基于两个排序索引列表进行对比评估。
        """
        original_metrics = self.evaluate_order(original_sorted_indices, relevance_scores, k_values)
        corrected_metrics = self.evaluate_order(corrected_sorted_indices, relevance_scores, k_values)

        improvements = {}
        for metric in original_metrics:
            if metric in corrected_metrics:
                improvements[f"{metric}_improvement"] = corrected_metrics[metric] - original_metrics[metric]

        return {
            'original': original_metrics,
            'corrected': corrected_metrics,
            'improvements': improvements
        }
    
    def compare_rankings(self, original_logits: List[float], corrected_logits: List[float], 
                        relevance_scores: List[int], k_values: Optional[List[int]] = None) -> Dict[str, Dict[str, float]]:
        """
        比较两个排序的性能
        
        Args:
            original_logits: 原始排序的logits
            corrected_logits: 校正后排序的logits
            relevance_scores: 相关性分数列表
            k_values: 评估的k值列表
            
        Returns:
            包含原始、校正和改进指标的字典
        """
        # 评估原始排序
        original_metrics = self.evaluate_ranking(original_logits, relevance_scores, k_values)
        
        # 评估校正后排序
        corrected_metrics = self.evaluate_ranking(corrected_logits, relevance_scores, k_values)
        
        # 计算改进
        improvements = {}
        for metric in original_metrics:
            if metric in corrected_metrics:
                improvements[f'{metric}_improvement'] = corrected_metrics[metric] - original_metrics[metric]
        
        return {
            'original': original_metrics,
            'corrected': corrected_metrics,
            'improvements': improvements
        }
    
    def statistical_significance_test(self, original_scores: List[float], corrected_scores: List[float]) -> Dict[str, float]:
        """
        统计显著性测试
        
        Args:
            original_scores: 原始分数列表
            corrected_scores: 校正后分数列表
            
        Returns:
            统计测试结果字典
        """
        try:
            from scipy import stats
            
            # 配对t检验
            t_stat, p_value = stats.ttest_rel(corrected_scores, original_scores)
            
            return {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
                'effect_size': float(np.mean(corrected_scores) - np.mean(original_scores))
            }
        except ImportError:
            # 如果没有scipy，使用简单的比较
            return {
                'mean_improvement': float(np.mean(corrected_scores) - np.mean(original_scores)),
                'std_improvement': float(np.std(np.array(corrected_scores) - np.array(original_scores))),
                'significant': False
            }
    
    def generate_evaluation_report(self, results: List[Dict]) -> Dict[str, any]:
        """
        生成评估报告
        
        Args:
            results: 评估结果列表
            
        Returns:
            评估报告字典
        """
        if not results:
            return {}
        
        # 提取所有改进指标
        all_improvements = defaultdict(list)
        for result in results:
            if 'performance_improvement' in result:
                for metric, value in result['performance_improvement'].items():
                    all_improvements[metric].append(value)
        
        # 计算统计信息
        report = {
            'total_queries': len(results),
            'metrics_summary': {}
        }
        
        for metric, values in all_improvements.items():
            if values:
                report['metrics_summary'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'positive_improvements': int(sum(1 for v in values if v > 0)),
                    'negative_improvements': int(sum(1 for v in values if v < 0))
                }
        
        return report
    
    def get_available_measures(self) -> List[str]:
        """
        获取可用的评估指标列表
        
        Returns:
            可用指标名称列表
        """
        return [str(measure) for measure in self.default_measures]
    
    def add_custom_measures(self, measures: List):
        """
        添加自定义评估指标
        
        Args:
            measures: 要添加的ir-measures指标列表
        """
        self.default_measures.extend(measures)
        print(f"✅ 添加自定义指标: {[str(m) for m in measures]}")
    
    def evaluate_with_custom_measures(self, logits: List[float], relevance_scores: List[int], 
                                    custom_measures: List) -> Dict[str, float]:
        """
        使用自定义指标评估排序性能
        
        Args:
            logits: 模型输出的logits分数
            relevance_scores: 相关性分数列表
            custom_measures: 自定义评估指标列表
            
        Returns:
            评估指标字典
        """
        # 准备数据
        qrels = self._prepare_qrels(relevance_scores)
        run = self._prepare_run(logits)

        # 计算指标
        results = ir_measures.calc_aggregate(custom_measures, qrels, run)
        
        # 转换为字典格式
        metrics = {}
        for measure, value in results.items():
            metrics[str(measure)] = float(value)
        
        return metrics

def create_ir_metrics_wrapper(k_values: List[int] = [1, 3, 5, 10]) -> IRMetricsWrapper:
    """
    创建IR评估指标包装器
    
    Args:
        k_values: 评估的k值列表
        
    Returns:
        IRMetricsWrapper实例
    """
    return IRMetricsWrapper(k_values)

# 为了向后兼容，保留原有的函数名
def create_ir_metrics(k_values: List[int] = [1, 3, 5, 10]) -> IRMetricsWrapper:
    """向后兼容的创建函数"""
    return create_ir_metrics_wrapper(k_values)

if __name__ == "__main__":
    # 测试 compare_orders
    original_sorted_indices = [4, 2, 1, 9, 6, 5, 7, 8, 3]
    corrected_sorted_indices =[4, 2, 1, 9, 6, 5, 7, 8, 3]
    relevance_scores = [2, 0, 0, 1, 0, 0, 0, 0, 0]
    ir_metrics = create_ir_metrics([1,3,5,9])
    comparison_result = ir_metrics.compare_orders(original_sorted_indices, corrected_sorted_indices, relevance_scores)
    # 按照表格打印
    from pprint import pprint
    pprint(comparison_result)