#!/usr/bin/env python3
"""Main entry point for the cleaned Qwen TREC-COVID / TREC-DL experiments."""
from __future__ import annotations

import os
import random
import sys
import traceback
import argparse
from typing import Any, Dict, List, Optional
from pathlib import Path

DATA_ROOT = Path(os.getenv("RERANK_BIAS_DATA_ROOT", "data")).expanduser()
os.environ.setdefault("HF_HOME", os.getenv("RERANK_BIAS_HF_HOME", str(DATA_ROOT / "huggingface")))
cwd = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, str(cwd))

try:
    from main.arguments import args
except ImportError:
    # 如果直接运行，使用相对导入
    from arguments import args

from src.logger import get_logger
logger = get_logger(__name__, log_config=getattr(args, "log", None))

from src.ir_metrics_wrapper import create_ir_metrics
from src.llm_utils import LLMExp_FixedBias, LLMExp_AdaptiveBias
from src.util import set_seed
from dataloader.factory import DataLoaderFactory

experiment_classes = {
        "fixed": LLMExp_FixedBias,
        "adaptive": LLMExp_AdaptiveBias,
    }


def data_path(*parts: str) -> str:
    """Build data path under RERANK_BIAS_DATA_ROOT."""
    return str(DATA_ROOT.joinpath(*parts))

def parse_args() -> None:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run the cleaned Qwen experiments on TREC-COVID or TREC-DL."
    )

    # 数据集参数
    parser.add_argument("--dataset_type", type=str, default="beir",
                       choices=["beir", "trec_dl"],
                       help="Dataset family.")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="BEIR 数据集名称（如 trec-covid）")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="数据目录路径（如果为 None，将根据 dataset_name 自动生成）")
    
    parser.add_argument("--same_passage_text", type=str, default="This is a placeholder passage for bias testing",
                       help="BEIR 数据集相同 passage 文本")
    
    # TREC DL 数据集特定参数
    parser.add_argument("--trec_dl_dataset", type=str, default="both",
                       choices=["dl19", "dl20", "dl21", "dl22", "dl23", "both", "all"],
                       help="TREC-DL 数据集（dl19, dl20, dl21, dl22, dl23, both, 或 all）")
    
    # BM25 粗排参数
    parser.add_argument("--use_bm25_retrieval", action="store_true",
                       help="使用 BM25 进行粗排")
    parser.add_argument("--bm25_top_k", type=int, default=100,
                       help="BM25 粗排的 top-k 值")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B",
                       help="模型名称")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="设备")
    parser.add_argument("--max_length", type=int, default=8192,
                       help="最大序列长度")
    parser.add_argument("--cache_use", action="store_true", default=True,
                       help="启用 KV cache")
    parser.add_argument("--no_cache_use", dest="cache_use", action="store_false",
                       help="关闭 KV cache")
    
    # 实验参数
    parser.add_argument("--num_queries", type=int, default=50,
                       help="查询数量")
    parser.add_argument("--passages_per_query", type=int, default=20,
                       help="Number of passages in each listwise ranking instance.")
    parser.add_argument("--bias_rate", type=float, default=1.0,
                       help="Calibration strength.")
    parser.add_argument("--experiment_type", type=str, default="fixed",
                       choices=list(experiment_classes.keys()),
                       help="实验类型：fixed=固定校正强度，adaptive=基于不确定性的自适应校正")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    # 输出参数
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录")
    parser.add_argument("--dry_run", action="store_true",
                       help="dry run 模式（仅加载数据，不运行模型）")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="详细输出")

    parsed = parser.parse_args()

    # 更新 args
    args.dataset_type = parsed.dataset_type
    args.dataset_name = parsed.dataset_name
    args.data_dir = parsed.data_dir
    
    # TREC DL 参数
    args.trec_dl_dataset = parsed.trec_dl_dataset
    
    # BM25 参数
    args.use_bm25_retrieval = parsed.use_bm25_retrieval
    args.bm25_top_k = parsed.bm25_top_k
    
    # 模型参数
    args.model.model_name = parsed.model_name
    args.model.device = parsed.device
    args.model.max_length = parsed.max_length
    args.model.cache_use = parsed.cache_use
    
    # 实验参数
    args.experiment.num_queries = parsed.num_queries
    args.experiment.passages_per_query = parsed.passages_per_query
    args.experiment.bias_rate = parsed.bias_rate
    args.experiment.experiment_type = parsed.experiment_type
    args.experiment.seed = parsed.seed
    args.experiment.same_passage_text = parsed.same_passage_text
    
    # 输出参数
    if parsed.output_dir:
        args.data.output_dir = Path(parsed.output_dir)
    
    args.dry_run = parsed.dry_run
    args.verbose = parsed.verbose
    
    # 重新初始化以应用所有更改
    args.__post_init__()

def load_test_data() -> List[Any]:
    """加载测试数据，自动识别数据集类型。"""
    logger.info(f"📊 加载数据集: {args.dataset_type} / {args.dataset_name}")
    
    try:
        # 准备数据加载器参数
        loader_kwargs = {
            "use_bm25_retrieval": args.use_bm25_retrieval,
            "bm25_top_k": args.bm25_top_k,
        }
        
        queries = []
        
        # 根据数据集类型加载数据
        if args.dataset_type == "trec_dl":
            # TREC DL 数据集
            if args.trec_dl_dataset in ("dl19", "dl20"):
                # TREC DL 2019/2020 使用 trec_dl 加载器
                loader = DataLoaderFactory.create(
                    data_dir=args.data_dir,
                    dataset_name=args.trec_dl_dataset,  # 传递数据集名称以便自动查找 BM25 文件
                    loader_type="trec_dl",
                    **loader_kwargs
                )
                queries = loader.load_all_data()
            elif args.trec_dl_dataset in ("dl21", "dl22", "dl23"):
                # TREC DL 2021/2022/2023 使用 trec_dl_212223 加载器
                # 根据数据集名称确定数据目录
                # dl21 -> trec21, dl22 -> trec22, dl23 -> trec23
                year = args.trec_dl_dataset[-2:]  # 获取最后两位数字
                data_dir = data_path("trec", f"trec{year}")
                
                loader = DataLoaderFactory.create(
                    data_dir=data_dir,
                    dataset_name=args.trec_dl_dataset,
                    loader_type="trec_dl_212223",
                    **loader_kwargs
                )
                queries = loader.load_all_data()
            elif args.trec_dl_dataset == "all":
                # 加载所有 TREC DL 数据集
                for dataset_key in ["dl19", "dl20"]:
                    data_dir = data_path(f"trec-dl-{dataset_key[-2:]}")
                    loader = DataLoaderFactory.create(
                        data_dir=data_dir,
                        dataset_name=dataset_key,
                        loader_type="trec_dl",
                        **loader_kwargs
                    )
                    queries.extend(loader.load_all_data())
                for dataset_key in ["dl21", "dl22", "dl23"]:
                    year = dataset_key[-2:]  # 获取最后两位数字
                    data_dir = data_path("trec", f"trec{year}")
                    loader = DataLoaderFactory.create(
                        data_dir=data_dir,
                        dataset_name=dataset_key,
                        loader_type="trec_dl_212223",
                        **loader_kwargs
                    )
                    queries.extend(loader.load_all_data())
            else:  # both (默认只包含 dl19 和 dl20)
                for dataset_key in ["dl19", "dl20"]:
                    data_dir = data_path(f"trec-dl-{dataset_key[-2:]}")
                    loader = DataLoaderFactory.create(
                        data_dir=data_dir,
                        dataset_name=dataset_key,  # 传递数据集名称以便自动查找 BM25 文件
                        loader_type="trec_dl",
                        **loader_kwargs
                    )
                    queries.extend(loader.load_all_data())
        
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
            return []
        
        # 如果查询数量超过限制，随机采样
        if len(queries) > args.experiment.num_queries:
            queries = random.sample(queries, args.experiment.num_queries)
            logger.info(f"📊 随机采样 {args.experiment.num_queries} 个查询（共 {len(queries)} 个）")
        
        logger.info(f"✅ 成功加载 {len(queries)} 个查询")
        return queries
    
    except Exception as exc:
        logger.error(f"❌ 数据加载失败: {exc}", exc_info=True)
        traceback.print_exc()
        return []

def main() -> None:
    """主函数。"""
    parse_args()
    
    global logger
    logger = get_logger(__name__, log_config=getattr(args, "log", None))
    
    args.configure_args(logger)
    set_seed(args.experiment.seed)
    
    if args.dry_run:
        logger.info("🧪 dry-run 模式：仅加载数据，不运行模型")
        queries = load_test_data()
        logger.info(f"成功加载 {len(queries)} 条查询")
        if queries:
            stats = queries[0].__class__.__module__ if hasattr(queries[0], '__class__') else "Unknown"
            logger.info(f"查询类型: {stats}")
        return
    
    logger.info("📊 初始化 IR 评估指标...")
    k_values = [1, 3, 5, min(10, args.experiment.passages_per_query)]
    ir_metrics = create_ir_metrics(k_values)
    
    exp_class = experiment_classes.get(
        getattr(args.experiment, "experiment_type", "fixed"),
        LLMExp_FixedBias,
    )
    experiment = exp_class(args)
    experiment.load_model()
    
    queries = load_test_data()
    if not queries:
        logger.error("❌ 数据加载失败，退出")
        return
    
    results: List[Dict[str, Any]] = []
    for idx, query in enumerate(queries, start=1):
        logger.info(f"📝 处理查询 {idx}/{len(queries)}: {query.query_id}")
        result = experiment.process_single_query(query, ir_metrics)
        results.append(result)
    
    experiment.save_results(results, args.data.output_dir, ir_metrics)
    logger.info(f"🎉 实验完成！结果保存在 {args.data.output_dir}")


if __name__ == "__main__":
    main()
