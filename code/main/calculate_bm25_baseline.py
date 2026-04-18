#!/usr/bin/env python3
"""Compute BM25 baselines for the cleaned TREC-COVID / TREC-DL setup."""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

DATA_ROOT = Path(os.getenv("RERANK_BIAS_DATA_ROOT", "data")).expanduser()
os.environ.setdefault("HF_HOME", os.getenv("RERANK_BIAS_HF_HOME", str(DATA_ROOT / "huggingface")))
cwd = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, str(cwd))

from main.arguments import args
from src.logger import get_logger
from src.ir_metrics_wrapper import create_ir_metrics
from dataloader.factory import DataLoaderFactory

logger = get_logger(__name__, log_config=getattr(args, "log", None))


def data_path(*parts: str) -> str:
    """Build data path under RERANK_BIAS_DATA_ROOT."""
    return str(DATA_ROOT.joinpath(*parts))


def calculate_bm25_baseline(
    dataset_type: str = "beir",
    dataset_name: str = None,
    data_dir: str = None,
    num_queries: int = None,
    bm25_top_k: int = 100,
    output_dir: str = None,
    trec_dl_dataset: str = None,
) -> Dict[str, Any]:
    """
    计算BM25的baseline性能指标
    
    Args:
        dataset_type: 数据集类型
        dataset_name: 数据集名称
        data_dir: 数据目录
        num_queries: 查询数量（None表示使用所有查询）
        bm25_top_k: BM25检索的top-k值
        output_dir: 输出目录
        
    Returns:
        包含BM25性能指标的字典
    """
    logger.info("=" * 80)
    logger.info("📊 计算BM25 Baseline性能指标")
    logger.info("=" * 80)
    
    # 配置参数
    args.dataset_type = dataset_type
    args.dataset_name = dataset_name
    args.data_dir = data_dir
    args.use_bm25_retrieval = True  # 强制使用BM25
    args.bm25_top_k = bm25_top_k
    
    if num_queries:
        args.experiment.num_queries = num_queries
    
    if trec_dl_dataset:
        args.trec_dl_dataset = trec_dl_dataset
    
    if output_dir:
        args.data.output_dir = Path(output_dir)
    
    args.__post_init__()
    
    logger.info(f"数据集类型: {args.dataset_type}")
    logger.info(f"数据集名称: {args.dataset_name}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"BM25 Top-K: {args.bm25_top_k}")
    logger.info(f"查询数量: {args.experiment.num_queries if num_queries else '全部'}")
    
    # 加载数据（使用BM25检索）
    logger.info("\n📥 加载数据（使用BM25检索）...")
    loader_kwargs = {
        "use_bm25_retrieval": True,
        "bm25_top_k": bm25_top_k,
    }
    
    queries = []
    
    # 根据数据集类型加载数据
    if args.dataset_type == "trec_dl":
        # 定义所有支持的TREC-DL数据集版本
        all_trec_dl_datasets = ["dl19", "dl20", "dl21", "dl22", "dl23"]
        
        if args.trec_dl_dataset in all_trec_dl_datasets:
            # 单个数据集版本
            dataset_key = args.trec_dl_dataset
            if dataset_key == "dl21" or dataset_key == "dl22" or dataset_key == "dl23":
                data_dir = data_path("trec", f"trec{dataset_key[-2:]}")
                loader = DataLoaderFactory.create(
                    data_dir=data_dir,
                    dataset_name=dataset_key,
                    loader_type="trec_dl_212223",
                    **loader_kwargs
                )
                queries.extend(loader.load_all_data())
            else:
                data_dir = data_path(f"trec-dl-{dataset_key[-2:]}")
                loader = DataLoaderFactory.create(
                    data_dir=data_dir,
                    dataset_name=dataset_key,
                    loader_type="trec_dl",
                    **loader_kwargs
                )
                queries.extend(loader.load_all_data())
        elif args.trec_dl_dataset == "both":
            # both: dl19 和 dl20
            for dataset_key in ["dl19", "dl20"]:
                data_dir = data_path(f"trec-dl-{dataset_key[-2:]}")
                loader = DataLoaderFactory.create(
                    data_dir=data_dir,
                    dataset_name=dataset_key,
                    loader_type="trec_dl",
                    **loader_kwargs
                )
                queries.extend(loader.load_all_data())
        elif args.trec_dl_dataset == "all":
            # all: 所有版本 (dl19, dl20, dl21, dl22, dl23)
            for dataset_key in all_trec_dl_datasets:
                if dataset_key == "dl21" or dataset_key == "dl22" or dataset_key == "dl23":
                    data_dir = data_path("trec", f"trec{dataset_key[-2:]}")
                    loader = DataLoaderFactory.create(
                        data_dir=data_dir,
                        dataset_name=dataset_key,
                        loader_type="trec_dl_212223",
                        **loader_kwargs
                    )
                    queries.extend(loader.load_all_data())
                elif dataset_key == "dl19" or dataset_key == "dl20":
                    data_dir = data_path(f"trec-dl-{dataset_key[-2:]}")
                    loader = DataLoaderFactory.create(
                        data_dir=data_dir,
                        dataset_name=dataset_key,
                        loader_type="trec_dl",
                        **loader_kwargs
                    )
                    queries.extend(loader.load_all_data())
        else:
            # 默认情况：如果没有指定或无效，尝试使用单个数据集
            if args.trec_dl_dataset:
                logger.warning(f"未知的TREC-DL数据集版本: {args.trec_dl_dataset}，尝试直接加载")
            loader = DataLoaderFactory.create(
                data_dir=args.data_dir,
                dataset_name=args.trec_dl_dataset,
                loader_type="trec_dl",
                **loader_kwargs
            )
            queries = loader.load_all_data()
    
    elif args.dataset_type == "beir":
        loader = DataLoaderFactory.create(
            data_dir=args.data_dir,
            dataset_name=args.dataset_name,
            loader_type="beir",
            **loader_kwargs
        )
        queries = loader.load_all_data()
    
    else:
        raise ValueError(f"Unsupported dataset_type in the cleaned repository: {args.dataset_type}")
    
    if not queries:
        logger.error("❌ 没有加载到测试数据")
        return {}
    
    # 限制查询数量
    if num_queries and len(queries) > num_queries:
        import random
        random.seed(42)
        queries = random.sample(queries, num_queries)
        logger.info(f"📊 随机采样 {num_queries} 个查询（共 {len(queries)} 个）")
    
    logger.info(f"✅ 成功加载 {len(queries)} 个查询")
    
    # 初始化IR评估指标
    k_values = [1, 3, 5, 10]
    ir_metrics = create_ir_metrics(k_values)
    
    # 计算BM25的baseline性能
    logger.info("\n📈 计算BM25 Baseline性能指标...")
    
    all_results = []
    all_zero_queries = []  # 记录全部相关性分数为0的查询
    total_queries = len(queries)  # 总查询数
    
    for idx, query in enumerate(queries, start=1):
        if idx % 10 == 0:
            logger.info(f"处理查询 {idx}/{len(queries)}: {query.query_id}")
        
        # 当使用BM25检索时：
        # 1. passages和passage_ids已经按照BM25分数降序排列（top100）
        # 2. relevance_scores存储的是从qrels中获取的真实相关性分数（不是BM25分数）
        # 3. 原始顺序（original_positions）就是BM25排序后的顺序
        
        # 获取top100的真实相关性分数（从qrels中获取，已存储在relevance_scores中）
        true_relevance = list(query.relevance_scores)
        
        # 检查top100中是否有非零相关性分数的文档
        non_zero_indices = [i for i, score in enumerate(true_relevance) if score > 0]
        zero_count = len(true_relevance) - len(non_zero_indices)
        
        # 如果top100中所有文档的相关性分数都是0，记录并跳过
        if len(non_zero_indices) == 0:
            all_zero_queries.append(query.query_id)
            logger.debug(f"查询 {query.query_id}: top{bm25_top_k}中所有{len(true_relevance)}个文档的相关性分数都是0，跳过")
            continue
        
        # 只保留非零分数的文档（从top100中过滤）
        filtered_relevance = [true_relevance[i] for i in non_zero_indices]
        filtered_indices = list(range(len(filtered_relevance)))
        
        # 记录过滤信息
        if zero_count > 0:
            logger.debug(f"查询 {query.query_id}: top{bm25_top_k}中有{zero_count}个文档相关性分数为0，已过滤，剩余{len(filtered_relevance)}个文档")
        
        # 计算IR指标（BM25排序 vs BM25排序，用于获取baseline性能）
        try:
            # 使用evaluate_order方法直接评估BM25排序
            metrics = ir_metrics.evaluate_order(
                filtered_indices,
                filtered_relevance
            )
            
            result = {
                'query_id': query.query_id,
                'metrics': metrics
            }
            all_results.append(result)
            
        except Exception as e:
            logger.warning(f"查询 {query.query_id} 评估失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue
    
    if not all_results:
        logger.error("❌ 没有成功评估的查询")
        return {}
    
    # 汇总结果
    logger.info("\n📊 汇总BM25 Baseline性能指标...")
    logger.info(f"总查询数: {total_queries}")
    logger.info(f"top{bm25_top_k}中全部相关性分数为0的查询数: {len(all_zero_queries)}")
    logger.info(f"成功评估的查询数: {len(all_results)}")
    if all_zero_queries:
        logger.info(f"top{bm25_top_k}中全部为0的查询ID（前10个）: {all_zero_queries[:10]}")
        if len(all_zero_queries) > 10:
            logger.info(f"... 还有 {len(all_zero_queries) - 10} 个查询")
    
    # 收集所有指标值
    metric_values = {}
    for result in all_results:
        metrics = result['metrics']
        # metrics是Dict[str, float]格式
        if isinstance(metrics, dict):
            for metric_name, metric_value in metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(float(metric_value))
        else:
            logger.warning(f"查询 {result['query_id']} 的metrics格式异常: {type(metrics)}")
    
    # 计算统计信息
    summary = {}
    for metric_name, values in metric_values.items():
        summary[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
        }
    
    # 输出结果
    logger.info("\n" + "=" * 80)
    logger.info("📊 BM25 Baseline性能指标汇总")
    logger.info("=" * 80)
    logger.info(f"总查询数: {total_queries}")
    logger.info(f"top{bm25_top_k}中全部相关性分数为0的查询数: {len(all_zero_queries)} (已删除)")
    logger.info(f"成功评估的查询数: {len(all_results)}")
    logger.info(f"BM25 Top-K: {bm25_top_k}")
    logger.info("\n指标详情:")
    
    # 按指标类型分组显示
    metric_groups = {
        '整体指标': ['AP', 'RR'],
        'NDCG指标': [m for m in metric_values.keys() if 'NDCG' in m],
        'Precision指标': [m for m in metric_values.keys() if m.startswith('P@')],
    }
    
    for group_name, metrics in metric_groups.items():
        if metrics:
            logger.info(f"\n{group_name}:")
            for metric_name in sorted(metrics):
                if metric_name in summary:
                    stats = summary[metric_name]
                    logger.info(
                        f"  {metric_name:15s}: "
                        f"均值={stats['mean']:.4f}, "
                        f"标准差={stats['std']:.4f}, "
                        f"中位数={stats['median']:.4f}"
                    )
    
    # 保存结果到CSV
    if args.data.output_dir:
        output_dir = Path(args.data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存汇总结果
        csv_path = output_dir / "bm25_baseline_summary.csv"
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['指标名称', '均值', '标准差', '最小值', '最大值', '中位数'])
            for metric_name in sorted(summary.keys()):
                stats = summary[metric_name]
                writer.writerow([
                    metric_name,
                    f"{stats['mean']:.6f}",
                    f"{stats['std']:.6f}",
                    f"{stats['min']:.6f}",
                    f"{stats['max']:.6f}",
                    f"{stats['median']:.6f}",
                ])
        
        logger.info(f"\n✅ 结果已保存到: {csv_path}")
        
        # 保存统计信息
        stats_path = output_dir / "bm25_baseline_statistics.csv"
        with open(stats_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['统计项', '数值'])
            writer.writerow(['总查询数', total_queries])
            writer.writerow([f'top{bm25_top_k}中全部相关性分数为0的查询数（已删除）', len(all_zero_queries)])
            writer.writerow(['成功评估的查询数', len(all_results)])
            writer.writerow(['BM25 Top-K', bm25_top_k])
        
        logger.info(f"✅ 统计信息已保存到: {stats_path}")
        
        # 保存全部为0的查询ID列表（top100中全部相关性分数为0）
        if all_zero_queries:
            zero_queries_path = output_dir / "all_zero_queries.csv"
            with open(zero_queries_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['query_id', '说明'])
                for query_id in all_zero_queries:
                    writer.writerow([query_id, f'top{bm25_top_k}中全部相关性分数为0'])
            logger.info(f"✅ top{bm25_top_k}中全部为0的查询ID列表已保存到: {zero_queries_path}")
        
        # 保存详细结果
        detailed_path = output_dir / "bm25_baseline_detailed.csv"
        with open(detailed_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            header = ['query_id'] + sorted(metric_values.keys())
            writer.writerow(header)
            # 写入数据
            for result in all_results:
                row = [result['query_id']]
                for metric_name in sorted(metric_values.keys()):
                    row.append(result['metrics'].get(metric_name, ''))
                writer.writerow(row)
        
        logger.info(f"✅ 详细结果已保存到: {detailed_path}")
    
    return {
        'summary': summary,
        'detailed_results': all_results,
        'total_queries': total_queries,
        'all_zero_queries': all_zero_queries,
        'all_zero_count': len(all_zero_queries),
        'num_queries': len(all_results),
        'bm25_top_k': bm25_top_k,
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Compute BM25 baselines for the cleaned repository.")
    
    # 数据集参数
    parser.add_argument("--dataset_type", type=str, default="beir",
                       choices=["beir", "trec_dl"],
                       help="数据集类型")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="数据集名称")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="数据目录路径")
    
    # TREC-DL特定参数
    parser.add_argument("--trec_dl_dataset", type=str, default=None,
                       choices=["dl19", "dl20", "dl21", "dl22", "dl23", "both", "all"],
                       help="TREC-DL数据集版本（dl19, dl20, dl21, dl22, dl23, both, 或 all）")
    
    # BM25参数
    parser.add_argument("--bm25_top_k", type=int, default=100,
                       help="BM25检索的top-k值")
    
    # 其他参数
    parser.add_argument("--num_queries", type=int, default=None,
                       help="查询数量（None表示使用所有查询）")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录")
    
    parsed = parser.parse_args()
    
    result = calculate_bm25_baseline(
        dataset_type=parsed.dataset_type,
        dataset_name=parsed.dataset_name,
        data_dir=parsed.data_dir,
        num_queries=parsed.num_queries,
        bm25_top_k=parsed.bm25_top_k,
        output_dir=parsed.output_dir,
        trec_dl_dataset=parsed.trec_dl_dataset,
    )
    
    if result:
        logger.info("\n🎉 BM25 Baseline计算完成！")
    else:
        logger.error("\n❌ BM25 Baseline计算失败！")


if __name__ == "__main__":
    main()
