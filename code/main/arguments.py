# arguments.py for the cleaned paper release
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Optional

try:
    from ..src.BaseConfig import ConfigBase, DataConfig, ExpConfig, LogConfig, ModelConfig
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    CODE_DIR = CURRENT_DIR.parent
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    from src.BaseConfig import ConfigBase, DataConfig, ExpConfig, LogConfig, ModelConfig


def _data_root() -> Path:
    """Resolve the data root from env, defaulting to ./data."""
    return Path(os.getenv("RERANK_BIAS_DATA_ROOT", "data")).expanduser()


def _data_path(*parts: str) -> str:
    """Build a path under the configured data root."""
    return str(_data_root().joinpath(*parts))


@dataclass
class Args(ConfigBase):
    """Configuration for the cleaned TREC-COVID / TREC-DL experiments."""

    # 数据集相关参数
    dataset_type: str = "beir"  # beir, trec_dl
    dataset_name: Optional[str] = None  # 数据集名称（如 trec-covid）
    data_dir: Optional[str] = None  # 数据目录路径（如果为 None，将根据 dataset_name 自动生成）
    
    # TREC DL 数据集特定参数
    trec_dl_dataset: str = "both"  # dl19, dl20, dl21, dl22, dl23, both, all
    
    # BM25 粗排参数
    use_bm25_retrieval: bool = False
    bm25_top_k: int = 100
    
    # 实验参数
    dry_run: bool = False
    verbose: bool = True

    data: DataConfig = field(
        default_factory=lambda: DataConfig(
            data_dir=str(_data_root()),
            output_dir=Path("results/main"),
            numbering_strategies=None,
        )
    )
    experiment: ExpConfig = field(
        default_factory=lambda: ExpConfig(
            num_queries=100,
            passages_per_query=20,
            bias_rate=1.0,
        )
    )
    log: LogConfig = field(default_factory=LogConfig)
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(model_name="Qwen/Qwen3-1.7B", device="auto", max_length=8192, cache_use=True)
    )

    def __post_init__(self) -> None:
        """初始化后处理，设置默认路径和目录。"""
        if isinstance(self.data.output_dir, str):
            self.data.output_dir = Path(self.data.output_dir)
        
        # 根据数据集类型设置默认路径
        # 设置数据目录
        if self.data_dir is None:
            self.data_dir = self._get_default_data_dir()
        
        self.data.data_dir = self.data_dir
        
        # 设置输出目录
        if self.data.output_dir == Path("results/main"):
            output_subdir = self.dataset_type
            if self.dataset_name:
                output_subdir = f"{output_subdir}/{self.dataset_name}"
            self.data.output_dir = Path(f"results/main/{output_subdir}")
        
        os.makedirs(self.data.output_dir, exist_ok=True)
        
    def _detect_dataset_type(self, dataset_name: str) -> str:
        """根据数据集名称自动检测数据集类型。"""
        try:
            from dataloader.factory import DataLoaderFactory
        except ImportError:
            # 如果导入失败，尝试从 code.dataloader 导入
            import sys
            from pathlib import Path
            CURRENT_DIR = Path(__file__).resolve().parent
            CODE_DIR = CURRENT_DIR.parent
            if str(CODE_DIR) not in sys.path:
                sys.path.insert(0, str(CODE_DIR))
            from dataloader.factory import DataLoaderFactory
        
        # TREC DL 数据集
        if dataset_name in DataLoaderFactory.TREC_DL_DATASETS or dataset_name in {"dl19", "dl20", "dl21", "dl22", "dl23"}:
            return "trec_dl"

        if dataset_name in DataLoaderFactory.BEIR_DATASETS:
            return "beir"

        raise ValueError(f"Unsupported dataset_name for the cleaned repository: {dataset_name}")
    
    def _get_default_data_dir(self) -> str:
        """根据数据集类型和名称获取默认数据目录。"""
        if self.dataset_type == "trec_dl":
            if self.trec_dl_dataset == "dl19":
                return _data_path("trec-dl-2019")
            elif self.trec_dl_dataset == "dl20":
                return _data_path("trec-dl-2020")
            elif self.trec_dl_dataset in {"dl21", "dl22", "dl23"}:
                return _data_path("trec", f"trec{self.trec_dl_dataset[-2:]}")
            else:
                return _data_path("trec-dl-2019")  # 默认
        elif self.dataset_type == "beir" and self.dataset_name:
            return _data_path("beir", self.dataset_name)
        else:
            return _data_path()

    def __str__(self) -> str:
        return pformat(self)


args = Args()
