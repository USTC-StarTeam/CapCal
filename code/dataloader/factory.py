"""Factory for the cleaned TREC-COVID / TREC-DL data loaders."""

from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    from src.abstract_dl import AbstractDataLoader, BaseQuery
except ImportError:
    import sys
    from pathlib import Path as PathLib
    CURRENT_DIR = PathLib(__file__).resolve().parent
    CODE_DIR = CURRENT_DIR.parent
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    from src.abstract_dl import AbstractDataLoader, BaseQuery
from .dl_19_20DataLoader import TestDataLoader
from .dl_21_22_23_loader import TREC212223DataLoader
from .beir_dataloader import BEIRDataLoader


class DataLoaderFactory:
    """Create loaders for the retained public experiment surface."""

    BEIR_DATASETS = ["trec-covid"]
    TREC_DL_DATASETS = [
        "dl19",
        "dl20",
        "dl21",
        "dl22",
        "dl23",
        "trec-dl-2019",
        "trec-dl-2020",
        "trec-dl-2021",
        "trec-dl-2022",
        "trec-dl-2023",
    ]
    
    @classmethod
    def create(
        cls,
        data_dir: str,
        dataset_name: Optional[str] = None,
        loader_type: Optional[str] = None,
        **kwargs
    ) -> AbstractDataLoader:
        """
        创建数据加载器
        
        Args:
            data_dir: 数据集目录路径
            dataset_name: 数据集名称（可选，用于自动识别类型）
            loader_type: 加载器类型（可选，强制指定类型）
                       可选值: 'beir', 'trec_dl', 'trec_dl_212223'
            **kwargs: 其他参数，传递给具体的加载器
        
        Returns:
            数据加载器实例
        
        Examples:
            >>> # 自动识别 BEIR 数据集
            >>> loader = DataLoaderFactory.create("data/beir/trec-covid")
            >>> 
            >>> # 指定 TREC DL 数据集
            >>> loader = DataLoaderFactory.create("data/trec-dl-2019", loader_type="trec_dl")
            >>> 
            >>> # 指定 BEIR 数据集名称
            >>> loader = DataLoaderFactory.create("data/beir", dataset_name="trec-covid")
        """
        # 如果指定了 loader_type，直接使用
        if loader_type:
            return cls._create_by_type(data_dir, loader_type, dataset_name, **kwargs)
        
        # 自动识别数据集类型
        loader_type = cls._detect_loader_type(data_dir, dataset_name)
        return cls._create_by_type(data_dir, loader_type, dataset_name, **kwargs)
    
    @classmethod
    def _detect_loader_type(cls, data_dir: str, dataset_name: Optional[str] = None) -> str:
        """
        自动检测数据加载器类型
        
        Args:
            data_dir: 数据集目录路径
            dataset_name: 数据集名称
        
        Returns:
            加载器类型字符串
        """
        data_path = Path(data_dir)
        
        # 检查是否是 TREC DL 2021/2022/2023 数据集
        if dataset_name in ['dl21', 'dl22', 'dl23', 'trec-dl-2021', 'trec-dl-2022', 'trec-dl-2023']:
            return 'trec_dl_212223'
        if 'trec21' in str(data_path).lower() or 'dl21' in str(data_path).lower() or '2021' in str(data_path).lower():
            # 检查是否是 trec21 目录
            if 'trec21' in str(data_path).name or '2021' in str(data_path).name:
                return 'trec_dl_212223'
        if 'trec22' in str(data_path).lower() or 'dl22' in str(data_path).lower() or '2022' in str(data_path).lower():
            if 'trec22' in str(data_path).name or '2022' in str(data_path).name:
                return 'trec_dl_212223'
        if 'trec23' in str(data_path).lower() or 'dl23' in str(data_path).lower() or '2023' in str(data_path).lower():
            if 'trec23' in str(data_path).name or '2023' in str(data_path).name:
                return 'trec_dl_212223'
        
        # 检查是否是 TREC DL 2019/2020 数据集
        if dataset_name in ['dl19', 'dl20', 'trec-dl-2019', 'trec-dl-2020']:
            return 'trec_dl'
        if 'trec-dl' in str(data_path).lower():
            # 检查是否是 dl19 或 dl20
            if 'dl19' in str(data_path).lower() or '2019' in str(data_path).lower():
                return 'trec_dl'
            elif 'dl20' in str(data_path).lower() or '2020' in str(data_path).lower():
                return 'trec_dl'
            # 其他 trec-dl 可能是 212223
            return 'trec_dl_212223'
        
        # 检查是否是 BEIR 数据集（使用统一的 BEIRDataLoader）
        if dataset_name in cls.BEIR_DATASETS:
            return 'beir'  # 使用统一的 BEIRDataLoader
        
        if 'beir' in str(data_path).lower():
            return 'beir'
        
        raise ValueError(f"Unsupported dataset path or name in the cleaned repository: {data_dir} / {dataset_name}")
    
    @classmethod
    def _create_by_type(
        cls,
        data_dir: str,
        loader_type: str,
        dataset_name: Optional[str] = None,
        **kwargs
    ) -> AbstractDataLoader:
        """
        根据类型创建数据加载器
        
        Args:
            data_dir: 数据集目录路径
            loader_type: 加载器类型
            dataset_name: 数据集名称
            **kwargs: 其他参数
        
        Returns:
            数据加载器实例（继承自 AbstractDataLoader）
        """
        if loader_type == 'beir':
            # 使用统一的 BEIRDataLoader（基于 beir 库）
            if not dataset_name:
                # 从路径中提取数据集名称
                dataset_name = Path(data_dir).name
            return BEIRDataLoader(
                data_dir=data_dir,
                dataset_name=dataset_name,
                **kwargs
            )
        if loader_type == 'trec_dl':
            # TREC DL 2019/2020 数据集
            # 从 data_dir 或 dataset_name 推断数据集名称
            trec_dataset_name = dataset_name
            if not trec_dataset_name:
                if 'dl19' in data_dir or '2019' in data_dir:
                    trec_dataset_name = 'dl19'
                elif 'dl20' in data_dir or '2020' in data_dir:
                    trec_dataset_name = 'dl20'
            
            return TestDataLoader(
                data_dir=data_dir,
                dataset_name=trec_dataset_name,
                **kwargs
            )
        
        if loader_type == 'trec_dl_212223':
            # TREC DL 2021/2022/2023 数据集
            # 从 data_dir 或 dataset_name 推断数据集名称
            trec_dataset_name = dataset_name
            if not trec_dataset_name:
                if 'dl21' in data_dir or 'trec21' in data_dir or '2021' in data_dir:
                    trec_dataset_name = 'dl21'
                elif 'dl22' in data_dir or 'trec22' in data_dir or '2022' in data_dir:
                    trec_dataset_name = 'dl22'
                elif 'dl23' in data_dir or 'trec23' in data_dir or '2023' in data_dir:
                    trec_dataset_name = 'dl23'
            
            return TREC212223DataLoader(
                data_dir=data_dir,
                dataset_name=trec_dataset_name,
                **kwargs
            )
        
        else:
            raise ValueError(
                f"Unknown loader_type: {loader_type}. "
                f"Supported types: beir, trec_dl, trec_dl_212223"
            )
    
    @classmethod
    def list_supported_datasets(cls) -> Dict[str, List[str]]:
        """
        列出所有支持的数据集
        
        Returns:
            字典，包含各类数据集的列表
        """
        return {
            'beir_datasets': cls.BEIR_DATASETS.copy(),
            'trec_dl_datasets': cls.TREC_DL_DATASETS.copy(),
        }
    
    @classmethod
    def load_dataset(
        cls,
        data_dir: str,
        dataset_name: Optional[str] = None,
        loader_type: Optional[str] = None,
        **kwargs
    ) -> List[BaseQuery]:
        """
        便捷方法：直接加载数据集并返回查询列表
        
        Args:
            data_dir: 数据集目录路径
            dataset_name: 数据集名称
            loader_type: 加载器类型
            **kwargs: 其他参数
        
        Returns:
            查询列表
        
        Examples:
            >>> queries = DataLoaderFactory.load_dataset("data/beir/trec-covid")
            >>> print(f"Loaded {len(queries)} queries")
        """
        loader = cls.create(data_dir, dataset_name, loader_type, **kwargs)
        return loader.load_all_data()
    
    @classmethod
    def get_statistics(
        cls,
        data_dir: str,
        dataset_name: Optional[str] = None,
        loader_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        便捷方法：获取数据集统计信息
        
        Args:
            data_dir: 数据集目录路径
            dataset_name: 数据集名称
            loader_type: 加载器类型
            **kwargs: 其他参数
        
        Returns:
            统计信息字典
        
        Examples:
            >>> stats = DataLoaderFactory.get_statistics("data/beir/trec-covid")
            >>> print(stats)
        """
        loader = cls.create(data_dir, dataset_name, loader_type, **kwargs)
        loader.load_all_data()
        return loader.get_query_statistics()


# 便捷函数
def create_loader(data_dir: str, dataset_name: Optional[str] = None, **kwargs):
    """便捷函数：创建数据加载器"""
    return DataLoaderFactory.create(data_dir, dataset_name, **kwargs)


def load_dataset(data_dir: str, dataset_name: Optional[str] = None, **kwargs):
    """便捷函数：直接加载数据集"""
    return DataLoaderFactory.load_dataset(data_dir, dataset_name, **kwargs)


def get_statistics(data_dir: str, dataset_name: Optional[str] = None, **kwargs):
    """便捷函数：获取数据集统计信息"""
    return DataLoaderFactory.get_statistics(data_dir, dataset_name, **kwargs)
