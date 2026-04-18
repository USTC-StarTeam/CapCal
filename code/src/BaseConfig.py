"""
公共模块出口。提供实验配置父类，供不同数据集/实验共用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Dict, Any
from pathlib import Path
from pprint import pformat
import datetime, logging, re

_BASE_DIR = Path(__file__).resolve().parents[2]#项目根目录

def _default_log_file(name: str = "bias_analysis") -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{name}_{timestamp}.log"

@dataclass
class LogConfig:
    """日志相关配置."""

    level: int = logging.INFO
    name: str = "bias_analysis"
    directory: Path = Path("log")
    file_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.file_name:
            self.file_name = _default_log_file(name=self.name)

    def resolve_path(self) -> Path:
        """获取日志文件的绝对路径."""
        file_name = self.file_name if self.file_name.endswith(".log") else f"{self.file_name}.log"
        log_path = (self.directory / file_name).expanduser()
        if not log_path.is_absolute():
            log_path = (_BASE_DIR / log_path).resolve()
        return log_path

@dataclass
class DataConfig:
    data_dir: str | List[str] | Dict[str, Any]#数据路径
    output_dir: Path = Path("results")#输出路径
    numbering_strategies: Optional[Sequence[str]] = None#编号策略

@dataclass
class ExpConfig:#实验配置
    num_queries: int #查询数量
    passages_per_query: int #每查询passage数量
    seed: int = 42 #随机种子
    bias_rate: float|List[float] =0.5 #bias校正强度 (0.1-2.0)
    experiment_type: str = "fixed"  # 实验类型（fixed / adaptive）
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])# metricConfig
    same_passage_text: str = "This is a placeholder passage for bias testing." #相同passage文本，占位符
    kwargs: Dict[str, Any] = field(default_factory=dict) #其他参数

@dataclass
class ModelConfig:#模型的基础配置
    # 模型配置
    model_name: str = "Qwen/Qwen3-1.7B"
    local_files_only: bool = False  # 允许直接从 Hugging Face 拉取公开 Qwen 模型
    device: str = "auto"  # auto / cuda / cpu
    max_length: int = 4096
    cache_use: bool = False  # 是否使用 KV cache 加速生成
    prompt_format:  str ="""
<|system|>
You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.
<|user|>
I will provide you with {passages_count} passages, each indicated by a numerical identifier [].
Rank the passages based on their relevance to the search query: {query}.
{passages_text}Search Query: {query}.
Rank the {passages_count} passages above based on their relevance to the search query.All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain.
<|assistant|>
"""
    

@dataclass
class ConfigBase:
    """实验相关基本配置的父类."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExpConfig = field(default_factory=ExpConfig)
    log: LogConfig = field(default_factory=LogConfig)

    verbose: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def configure_args(self,logger: logging.Logger=None):
        logger = self.__log_lode(logger)
        print(getattr(self, "data", None))
        args_info = f"""
        🚀 BEIR Rerank Bias 研究
        ============================
        📋 实验配置:
         - 数据集: {self.data.data_dir}
         - 查询数量: {self.experiment.num_queries}
         - 每查询passage数量: {self.experiment.passages_per_query}
         - 输出目录: {self.data.output_dir}
         - 校正强度: {self.experiment.bias_rate}
        ============================
        """
        logger.info(args_info)
        logger.info(pformat(self))

    def __log_lode(self,logger: logging.Logger=None):
        if logger is not None:
            return logger
        else:
            try:
                from code.src.logger import get_logger
                logger = get_logger(__name__, log_config=getattr(self, "log", None))
            except ModuleNotFoundError as e:
                from src.logger import get_logger
                logger = get_logger(__name__, log_config=getattr(self, "log", None))
            except:
                from logger import get_logger
                logger = get_logger(__name__, log_config=getattr(self, "log", None))
            return logger

# 必须要填写的配置
"""
data: DataConfig data_dir
experiment: ExpConfig num_queries, passages_per_query, bias_rate, kwargs
kwargs: Dict[str, Any] #可选参数，用来填坑
"""

__all__ = ["ConfigBase","DataConfig","ExpConfig","LogConfig","ModelConfig"]#导出ConfigBase类和便捷函数
