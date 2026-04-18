# dataloader abstract class
"""
通用数据加载器抽象基类，约束不同数据源的加载流程。
"""
from __future__ import annotations

import abc
import hashlib
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

@dataclass
class BaseQuery:
    """查询数据的统一结构，供下游评估复用。"""
    query_id: str #查询id
    query_text: str #查询文本
    passages: Sequence[str] #passage文本
    passage_ids: Sequence[str] #passage id
    relevance_scores: Sequence[int] #相关性得分
    original_positions: Sequence[int] #原始位置

class AbstractDataLoader(abc.ABC):
    """数据加载器父类，提供统一流程骨架。"""

    def __init__(
        self, 
        data_dir: str, 
        cache_path: Optional[str] = None, #缓存文件路径
        use_bm25_retrieval: bool = False, #是否使用BM25粗排
        bm25_top_k: int = 100, #BM25粗排的top-k值
    ):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据集目录路径
            cache_path: 缓存文件路径（如果为 None，自动生成）
            use_bm25_retrieval: 是否使用 BM25 进行粗排（如果为 True，将使用 BM25 检索 top-k 文档）
            bm25_top_k: BM25 检索返回的 top-k 文档数量（默认 100）
        """
        self.data_dir = os.path.abspath(data_dir)  # 规范化路径
        # 如果未提供 cache_path，则根据 data_dir 自动生成
        self.cache_path = self._generate_cache_path(use_bm25_retrieval, bm25_top_k) if cache_path is None else cache_path
        self.queries: Dict[str, str] = {} #查询文本
        self.corpus: Dict[str, Dict[str, Any]] = {} #语料库
        self.qrels: List[Dict[str, Any]] = [] #相关性判断列表
        self.test_queries: List[BaseQuery] = [] 
        
        # BM25 相关参数
        self.use_bm25_retrieval = use_bm25_retrieval
        self.bm25_top_k = bm25_top_k
        
        # BM25 索引（延迟初始化）
        self._bm25_index = None
        self._corpus_texts = []
        self._corpus_tokenized = []
        self._corpus_ids_list = []

    def _generate_cache_path(self, use_bm25: bool = False, bm25_top_k: int = 100) -> str: #根据 data_dir 自动生成缓存路径
        """根据 data_dir 自动生成缓存路径。
        
        缓存文件将保存在 data_dir 的父目录下的 cache 文件夹中，
        文件名基于 data_dir 的路径生成唯一标识。
        
        Args:
            use_bm25: 是否使用 BM25 粗排
            bm25_top_k: BM25 top-k 值
        """
        normalized_path = self.data_dir #规范化路径，self.data_dir 已经是绝对路径，直接使用
        path_hash = hashlib.md5(normalized_path.encode('utf-8')).hexdigest()[:8]# 使用路径的哈希值生成唯一文件名（避免路径过长或特殊字符问题）
    
        # 获取 data_dir 的目录名作为标识
        dir_name = os.path.basename(normalized_path) or "data"  # 如果为空则使用默认值
        
        # 在 data_dir 的父目录下创建 cache 文件夹
        parent_dir = os.path.dirname(normalized_path)
        cache_dir = os.path.join(parent_dir, 'cache')
        
        # 根据是否使用 BM25 生成不同的缓存文件名
        if use_bm25:
            cache_filename = f"{dir_name}_{path_hash}_bm25_{bm25_top_k}.pkl"
        else:
            cache_filename = f"{dir_name}_{path_hash}.pkl"
        cache_path = os.path.join(cache_dir, cache_filename)
        
        return cache_path

    def load_all_data(self, apply_bm25: Optional[bool] = None) -> List[BaseQuery]:
        """
        模板方法：按顺序执行各阶段加载。优先使用缓存。
        
        Args:
            apply_bm25: 是否应用 BM25 粗排（如果为 None，使用初始化时的 use_bm25_retrieval 设置）
        
        Returns:
            查询列表（如果启用 BM25，返回 BM25 粗排后的结果）
        """
        should_apply_bm25 = apply_bm25 if apply_bm25 is not None else self.use_bm25_retrieval #确定是否使用 BM25
        
        current_cache_path = self._generate_cache_path(should_apply_bm25, self.bm25_top_k) #根据当前的 BM25 设置重新生成缓存路径（确保 BM25 和非 BM25 的缓存分开）
        
        if current_cache_path != self.cache_path: #如果缓存路径发生变化，更新它
            self.cache_path = current_cache_path
        
        # 如果提供了缓存路径且缓存存在，直接加载缓存
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"✅ 检测到缓存文件，从 {self.cache_path} 加载数据...")
            self._load_cache()
            return self.test_queries
        
        # 否则执行正常加载流程
        if should_apply_bm25:
            print(f"未找到 BM25 缓存，开始加载原始数据并应用 BM25 粗排...")
        else:
            print("未找到缓存，开始加载原始数据...")
        
        self._load_queries()
        self._load_corpus()
        self._load_qrels()
        self._process_data()
        
        # 如果启用 BM25 且子类没有在 _process_data 中处理，则在这里应用
        if should_apply_bm25:
            if not self._has_bm25_applied():
                print(f"🔍 应用 BM25 粗排（top_k={self.bm25_top_k}）...")
                self._apply_bm25_retrieval()
        
        # 如果提供了缓存路径，保存缓存（限制为前100篇passages）
        if self.cache_path:
            self._save_cache()
        
        return self.test_queries

    @abc.abstractmethod
    def _load_queries(self) -> None:
        """加载查询文本."""

    @abc.abstractmethod
    def _load_corpus(self) -> None:
        """加载语料库."""

    @abc.abstractmethod
    def _load_qrels(self) -> None:
        """加载相关性判断."""

    @abc.abstractmethod
    def _process_data(self) -> None:
        """根据原始数据构建 `BaseQuery` 列表."""

    def _has_bm25_applied(self) -> bool:
        """
        检查子类是否已经在 _process_data 中应用了 BM25
        
        Returns:
            如果已经应用了 BM25，返回 True
        """
        # 子类可以重写此方法，如果已经在 _process_data 中处理了 BM25，返回 True
        # 默认返回 False，让基类应用 BM25
        return False
    
    def _apply_bm25_retrieval(self) -> None:
        """
        应用 BM25 粗排到已处理的查询结果
        
        这是一个可选的方法，子类可以重写以实现自定义的 BM25 逻辑。
        如果子类没有重写，将使用基类的通用实现。
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 library is required for BM25 retrieval. "
                "Install with: pip install rank-bm25"
            )
        
        if not self.test_queries:
            return
        
        # 构建 BM25 索引
        if self._bm25_index is None:
            self._build_bm25_index()
        
        # 构建 qrels 字典以便快速查找相关性分数
        qrels_dict = {}
        for qrel in self.qrels:
            query_id = qrel.get('query-id') or qrel.get('query_id')
            doc_id = qrel.get('corpus-id') or qrel.get('corpus_id') or qrel.get('doc_id')
            score = qrel.get('score', 0)
            if query_id and doc_id:
                if query_id not in qrels_dict:
                    qrels_dict[query_id] = {}
                qrels_dict[query_id][doc_id] = score
                
        # 对每个查询应用 BM25 检索
        new_test_queries = []
        for query in self.test_queries:
            # 使用 BM25 检索 top-k 文档
            bm25_results = self._retrieve_with_bm25(query.query_text, self.bm25_top_k)
            
            # 获取检索到的文档
            passages = []
            passage_ids = []
            relevance_scores = []
            
            # 获取该查询的 qrels（如果存在）
            query_qrels = qrels_dict.get(query.query_id, {})
            
            for doc_id, bm25_score, rank in bm25_results:
                if doc_id in self.corpus:
                    doc_data = self.corpus[doc_id]
                    passage_text = doc_data.get('text', '') if isinstance(doc_data, dict) else str(doc_data)
                    passages.append(passage_text)
                    passage_ids.append(doc_id)
                    # 尝试从 qrels 中获取真实的相关性分数，如果不存在则设为 0
                    relevance_score = query_qrels.get(doc_id, 0)
                    relevance_scores.append(relevance_score)
            
            if passages:
                # 创建新的查询对象，使用 BM25 检索结果
                new_query = BaseQuery(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    passages=passages,
                    passage_ids=passage_ids,
                    relevance_scores=relevance_scores,
                    original_positions=list(range(len(passages)))
                )
                new_test_queries.append(new_query)
        
        # 替换原有的查询列表
        self.test_queries = new_test_queries
        print(f"✅ BM25 粗排完成，处理了 {len(new_test_queries)} 个查询")
    
    def _build_bm25_index(self) -> None:
        """
        构建 BM25 索引
        
        子类可以重写此方法以实现自定义的索引构建逻辑。
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 library is required for BM25 retrieval. "
                "Install with: pip install rank-bm25"
            )
        
        print(f"🔨 构建 BM25 索引（语料库大小: {len(self.corpus)}）...")
        
        # 准备语料库文本和 tokenized 版本
        self._corpus_texts = []
        self._corpus_tokenized = []
        self._corpus_ids_list = []
        
        for doc_id, doc_data in self.corpus.items():
            # 处理不同的数据格式
            if isinstance(doc_data, dict):
                text = doc_data.get('text', '')
            else:
                text = str(doc_data)
            
            # 简单的 tokenization（按空格分割）
            tokens = text.lower().split()
            self._corpus_texts.append(text)
            self._corpus_tokenized.append(tokens)
            self._corpus_ids_list.append(doc_id)
        
        # 构建 BM25 索引
        self._bm25_index = BM25Okapi(self._corpus_tokenized)
        print(f"✅ BM25 索引构建完成")
    
    def _retrieve_with_bm25(self, query_text: str, top_k: int) -> List[tuple]:
        """
        使用 BM25 检索文档
        
        Args:
            query_text: 查询文本
            top_k: 返回的 top-k 文档数量
        
        Returns:
            List[tuple]: [(doc_id, score, rank), ...] 按分数降序排列
        """
        if self._bm25_index is None:
            self._build_bm25_index()
        
        # Tokenize 查询
        query_tokens = query_text.lower().split()
        
        # 获取 BM25 分数
        scores = self._bm25_index.get_scores(query_tokens)
        
        # 获取 top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc_id = self._corpus_ids_list[idx]
            score = float(scores[idx])
            results.append((doc_id, score, rank))
        
        return results

    def _save_cache(self) -> None:
        """保存缓存：存储所有query和前100篇passages。"""
        if not self.cache_path:
            return
        
        # 限制每个query的passages为前100篇
        cached_queries = []
        for query in self.test_queries:
            # 取前100篇passages及其对应的信息
            max_passages = min(100, len(query.passages))
            cached_query = BaseQuery(
                query_id=query.query_id,
                query_text=query.query_text,
                passages=list(query.passages[:max_passages]),
                passage_ids=list(query.passage_ids[:max_passages]),
                relevance_scores=list(query.relevance_scores[:max_passages]),
                original_positions=list(query.original_positions[:max_passages]),
            )
            cached_queries.append(cached_query)
        
        # 确保缓存目录存在
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        # 保存到pickle文件
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cached_queries, f)
        
        # 根据缓存路径判断是否使用了 BM25
        cache_type = "BM25 粗排" if "_bm25_" in os.path.basename(self.cache_path) else "原始数据"
        print(f"💾 {cache_type}缓存已保存到 {self.cache_path} (共 {len(cached_queries)} 个query，每个最多100篇passages)")

    def _load_cache(self) -> None:
        """从缓存文件加载数据。"""
        if not self.cache_path or not os.path.exists(self.cache_path):
            raise FileNotFoundError(f"缓存文件不存在: {self.cache_path}")
        
        with open(self.cache_path, 'rb') as f:
            self.test_queries = pickle.load(f)
        
        print(f"从缓存加载了 {len(self.test_queries)} 个query")

    def get_query_statistics(self) -> Dict[str, Any]:
        """给子类提供的基础统计，便于快速调试。"""
        if not self.test_queries:
            return {}

        total_queries = len(self.test_queries)
        total_passages = sum(len(query.passages) for query in self.test_queries)
        flat_scores = [score for query in self.test_queries for score in query.relevance_scores]

        return {
            "total_queries": total_queries,
            "total_passages": total_passages,
            "avg_passages_per_query": total_passages / total_queries if total_queries else 0,
            "total_judgments": len(flat_scores),
            "relevant_judgments": sum(score > 0 for score in flat_scores),
        }