"""Public data-loader surface for the cleaned paper release."""

# 统一工厂类（推荐使用）
from .factory import (
    DataLoaderFactory,
    create_loader,
    load_dataset,
    get_statistics
)

from .dl_19_20DataLoader import TestDataLoader, TestQuery
from .dl_21_22_23_loader import TREC212223DataLoader, TREC212223Query
from .beir_dataloader import BEIRDataLoader

# 从 abstract_dl 导入基础类
try:
    from ..src.abstract_dl import AbstractDataLoader, BaseQuery
except ImportError:
    import sys
    from pathlib import Path
    CURRENT_DIR = Path(__file__).resolve().parent
    CODE_DIR = CURRENT_DIR.parent
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    from src.abstract_dl import AbstractDataLoader, BaseQuery

__all__ = [
    # 统一工厂类（推荐）
    'DataLoaderFactory',
    'create_loader',
    'load_dataset',
    'get_statistics',
    # 基础类
    'AbstractDataLoader',
    'BaseQuery',
    # 数据加载器（都继承自 AbstractDataLoader）
    'TestDataLoader',
    'TestQuery',
    'TREC212223DataLoader',
    'TREC212223Query',
    'BEIRDataLoader',
]
