"""
TREC DL 2019 数据加载器
支持从本地目录加载TREC DL 2019测试集数据
支持从预下载的 BM25 数据文件读取
"""
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from src.abstract_dl import AbstractDataLoader, BaseQuery


@dataclass
class TestQuery(BaseQuery):
    """测试查询数据结构."""


class TestDataLoader(AbstractDataLoader):
    """TREC DL 2019 /2020测试集数据加载器"""

    def __init__(
        self, 
        data_dir: str,
        use_bm25_retrieval: bool = False,
        bm25_top_k: int = 100,
        bm25_data_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ):
        """
        初始化 TREC DL 数据加载器
        
        Args:
            data_dir: 数据集目录路径
            use_bm25_retrieval: 是否使用 BM25 进行粗排
            bm25_top_k: BM25 检索返回的 top-k 文档数量（默认 100）
            bm25_data_path: 预下载的 BM25 数据路径（可选）
            dataset_name: 数据集名称（dl19 或 dl20），用于自动查找 BM25 文件
        """
        super().__init__(
            data_dir,
            cache_path=None,  # 使用自动生成的缓存路径
            use_bm25_retrieval=use_bm25_retrieval,
            bm25_top_k=bm25_top_k
        )
        
        # 预下载的 BM25 数据路径
        self.bm25_data_path = bm25_data_path
        self.dataset_name = dataset_name
        
        # 尝试从 data_dir 推断数据集名称
        if not self.dataset_name:
            if 'dl19' in data_dir or '2019' in data_dir:
                self.dataset_name = 'dl19'
            elif 'dl20' in data_dir or '2020' in data_dir:
                self.dataset_name = 'dl20'
        
        # 优先查找 whybe-choi 数据集（本身就是 BM25 粗排后的，无论参数如何都使用）
        # 对于 dl19 和 dl20，直接使用 whybe-choi 数据集，不使用 beir-eval-bm25-top100
        self._use_whybe_choi_format = False
        if self.dataset_name and not self.bm25_data_path:
            try:
                current_dir = Path(__file__).resolve().parent
                code_dir = current_dir.parent
                base_data_dir = code_dir.parent / "data"
            except:
                base_data_dir = Path("data")
            
            # 查找 whybe-choi 数据集（dl19 和 dl20 的唯一数据源）
            whybe_choi_dir = base_data_dir / "datasets" / f"whybe-choi___trec-dl-{self.dataset_name[-2:]}"
            if whybe_choi_dir.exists():
                # 查找 test.arrow 文件
                test_arrow_pattern = whybe_choi_dir / "default" / "0.0.0" / "*" / f"trec-dl-{self.dataset_name[-2:]}-test.arrow"
                test_arrow_files = glob.glob(str(test_arrow_pattern))
                if test_arrow_files:
                    self.bm25_data_path = test_arrow_files[0]
                    self._use_whybe_choi_format = True
                    # 强制启用 BM25 检索（因为数据本身就是 BM25 粗排后的）
                    self.use_bm25_retrieval = True
                    print(f"📁 检测到 whybe-choi 数据集（BM25 粗排后），自动使用: {self.bm25_data_path}")
                else:
                    print(f"⚠️  whybe-choi 数据集目录存在但未找到 test.arrow 文件: {whybe_choi_dir}")
            else:
                print(f"⚠️  未找到 whybe-choi 数据集目录: {whybe_choi_dir}")
                print(f"   请确保已下载 whybe-choi/trec-dl-{self.dataset_name[-2:]} 数据集")
                # 对于 dl19 和 dl20，如果找不到 whybe-choi 数据集，禁用 BM25 粗排
                if self.dataset_name in ('dl19', 'dl20'):
                    self.use_bm25_retrieval = False
                    print(f"📋 对于 {self.dataset_name}，未找到 whybe-choi 数据集，将使用原始 qrels（不进行 BM25 粗排）")
        
        # 注意：对于 dl19 和 dl20，不再查找 beir-eval-bm25-top100 格式
        # 因为 whybe-choi 数据集是这些数据集的唯一数据源
        # 如果找不到 whybe-choi 数据集，禁用 BM25 粗排，使用原始 qrels
        if self.dataset_name in ('dl19', 'dl20') and not self.bm25_data_path:
            # 对于 dl19 和 dl20，如果没有 whybe-choi 数据集，禁用 BM25 粗排
            self.use_bm25_retrieval = False
            print(f"⚠️  对于 {self.dataset_name}，未找到 whybe-choi 数据集，将使用原始 qrels（不进行 BM25 粗排）")

    def load_all_data(self, apply_bm25=None) -> List[TestQuery]:
        """
        重写父类方法，如果使用预下载的 BM25 文件，跳过加载 corpus
        
        Args:
            apply_bm25: 是否应用 BM25 粗排（如果为 None，使用初始化时的 use_bm25_retrieval 设置）
        
        Returns:
            查询列表
        """
        should_apply_bm25 = apply_bm25 if apply_bm25 is not None else self.use_bm25_retrieval
        
        # 如果检测到 whybe-choi 数据集（本身就是 BM25 粗排后的），直接使用，无论参数如何
        # 或者如果使用 BM25 且有预下载的文件，优化加载流程
        if self.bm25_data_path and os.path.exists(self.bm25_data_path):
            if self._use_whybe_choi_format:
                print(f"📖 检测到 whybe-choi 数据集（BM25 粗排后），直接使用...")
            else:
                if not should_apply_bm25:
                    # 如果没有启用 BM25，使用父类的标准流程
                    queries = super().load_all_data(apply_bm25)
                    print(f"Successfully loaded {len(queries)} test queries")
                    return queries
                print(f"📖 检测到预下载的 BM25 文件，直接使用（跳过加载完整 corpus）...")
            
            # 检查缓存（但跳过父类的缓存检查，因为我们有自己的流程）
            current_cache_path = self._generate_cache_path(should_apply_bm25, self.bm25_top_k)
            if current_cache_path != self.cache_path:
                self.cache_path = current_cache_path
            
            # 如果缓存存在，直接加载
            if self.cache_path and os.path.exists(self.cache_path):
                print(f"✅ 检测到缓存文件，从 {self.cache_path} 加载数据...")
                self._load_cache()
                return self.test_queries
            
            # 加载查询和 qrels
            self._load_queries()
            self._load_qrels()
            
            # 如果是 whybe-choi 格式，需要加载 corpus（因为 Arrow 文件不包含文本）
            # 但我们可以先加载 BM25 数据，然后只加载涉及的文档（优化）
            if self._use_whybe_choi_format:
                # 先加载 BM25 数据，获取涉及的文档 ID
                bm25_data_temp = self._load_bm25_from_file()
                if bm25_data_temp:
                    # 收集所有涉及的文档 ID
                    needed_doc_ids = set()
                    for matched_data in bm25_data_temp.values():
                        for bm25_result in matched_data.get('bm25_results', []):
                            needed_doc_ids.add(bm25_result['doc_id'])
                    
                    print(f"📖 whybe-choi 格式：需要加载 {len(needed_doc_ids)} 个文档的文本...")
                    # 只加载需要的文档
                    self._load_corpus_selective(needed_doc_ids)
                    # 缓存 BM25 数据，避免在 _process_data_from_bm25_file 中重复加载
                    self._cached_bm25_data = bm25_data_temp
                else:
                    # 如果加载失败，回退到加载完整 corpus
                    print("⚠️  无法加载 BM25 数据，回退到加载完整 corpus...")
                    self._load_corpus()
                    self._cached_bm25_data = None
            else:
                # beir-eval-bm25-top100 格式不需要加载 corpus，文本已在文件中
                self._cached_bm25_data = None
            
            # 直接从 BM25 文件处理数据
            self._process_data_from_bm25_file()
            
            # 保存缓存
            if self.cache_path:
                self._save_cache()
            
            print(f"Successfully loaded {len(self.test_queries)} test queries from BM25 file")
            return self.test_queries
        
        # 否则使用父类的标准流程
        # 但对于 dl19 和 dl20，如果没有 whybe-choi 数据集，强制不使用 BM25
        if self.dataset_name in ('dl19', 'dl20') and not self.bm25_data_path:
            # 强制不使用 BM25，使用原始 qrels
            apply_bm25 = False
            print(f"📋 对于 {self.dataset_name}，使用原始 qrels（不进行 BM25 粗排）")
        
        queries = super().load_all_data(apply_bm25)
        print(f"Successfully loaded {len(queries)} test queries")
        return queries
    
    def _process_data_from_bm25_file(self):
        """直接从 BM25 文件处理数据，不需要加载 corpus"""
        print(f"🔍 从 BM25 文件直接处理数据（top_k={self.bm25_top_k}）...")
        
        # 标记：已经处理了 BM25
        self._bm25_handled_in_process = True
        
        # 构建 qrels 字典以便快速查找相关性分数
        qrels_dict = {}
        for qrel in self.qrels:
            query_id_key = qrel.get('query-id') or qrel.get('query_id')
            doc_id_key = qrel.get('corpus-id') or qrel.get('corpus_id') or qrel.get('doc_id')
            score = qrel.get('score', 0)
            if query_id_key and doc_id_key:
                if query_id_key not in qrels_dict:
                    qrels_dict[query_id_key] = {}
                qrels_dict[query_id_key][doc_id_key] = int(score)
        
        # 从文件加载 BM25 数据（如果已缓存则使用缓存）
        if hasattr(self, '_cached_bm25_data') and self._cached_bm25_data is not None:
            bm25_data = self._cached_bm25_data
        else:
            bm25_data = self._load_bm25_from_file()
        
        if not bm25_data:
            raise RuntimeError("无法从 BM25 文件加载数据")
        
        # 直接从 BM25 文件处理数据，而不是从 queries.jsonl 遍历
        processed_count = 0
        skipped_count = 0
        
        for query_id, matched_data in tqdm(bm25_data.items(), desc="处理 BM25 数据"):
            # 获取查询文本：优先使用 BM25 文件中的，否则从 queries 中获取
            query_text = matched_data.get('q_text', '')
            if not query_text and query_id in self.queries:
                query_text = self.queries[query_id]
            elif not query_text:
                # 如果都没有，跳过这个查询
                skipped_count += 1
                continue
            
            # 获取该查询的 BM25 结果和 qrels
            bm25_results = matched_data.get('bm25_results', [])
            file_qrels = matched_data.get('qrels', {})
            
            # 合并文件中的 qrels 和原始的 qrels_dict
            query_qrels = qrels_dict.get(query_id, {})
            # 文件中的 qrels 优先级更高
            query_qrels.update(file_qrels)
            
            # 获取passage文本（直接从 BM25 文件）
            passages = []
            valid_passage_ids = []
            valid_scores = []
            
            for bm25_result in bm25_results:
                doc_id = bm25_result['doc_id']
                
                # 如果 BM25 结果中包含文本（beir-eval-bm25-top100 格式）
                if 'text' in bm25_result and bm25_result['text']:
                    passage_text = bm25_result['text']
                    title = bm25_result.get('title', '')
                    if title:
                        passage_text = f"{title}\n{passage_text}".strip()
                # 如果是 whybe-choi 格式，从 corpus 获取文本
                elif self._use_whybe_choi_format and doc_id in self.corpus:
                    passage_text = self.corpus[doc_id]['text']
                    title = self.corpus[doc_id].get('title', '')
                    if title:
                        passage_text = f"{title}\n{passage_text}".strip()
                else:
                    # 如果都没有，跳过
                    continue  # 跳过没有文本的 passage
                
                passages.append(passage_text)
                valid_passage_ids.append(doc_id)
                # 尝试从 qrels 中获取真实的相关性分数，如果不存在则设为 0
                relevance_score = query_qrels.get(doc_id, 0)
                valid_scores.append(int(relevance_score))
            
            if passages:  # 只有当有有效passages时才创建TestQuery
                original_positions = list(range(len(passages)))
                
                test_query = TestQuery(
                    query_id=query_id,
                    query_text=query_text,
                    passages=passages,
                    passage_ids=valid_passage_ids,
                    relevance_scores=valid_scores,
                    original_positions=original_positions
                )
                self.test_queries.append(test_query)
                processed_count += 1
        
        print(f"Processed {processed_count} test queries from BM25 file")
        if skipped_count > 0:
            print(f"⚠️  跳过了 {skipped_count} 个没有查询文本的查询")

    def _load_queries(self):
        """加载查询数据"""
        queries_file = os.path.join(self.data_dir, "queries.jsonl")
        print(f"Loading queries from {queries_file}...")
        
        try:
            with open(queries_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    self.queries[data['_id']] = data['text']
            
            print(f"Loaded {len(self.queries)} queries")
        except Exception as e:
            print(f"Error loading queries: {e}")
            raise
    
    def _load_corpus(self):
        """加载语料库数据"""
        corpus_file = os.path.join(self.data_dir, "corpus.jsonl")
        print(f"Loading corpus from {corpus_file}...")

        try:
            # 先获得文件总行数以设置进度条总数
            with open(corpus_file, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading corpus", total=total_lines):
                    data = json.loads(line.strip())
                    self.corpus[data['_id']] = {
                        'title': data.get('title', ''),
                        'text': data['text']
                    }
            print(f"Loaded {len(self.corpus)} corpus entries")
        except Exception as e:
            print(f"Error loading corpus: {e}")
            raise
    
    def _load_corpus_selective(self, needed_doc_ids: set):
        """
        选择性加载语料库数据（只加载需要的文档）
        
        Args:
            needed_doc_ids: 需要加载的文档 ID 集合
        """
        corpus_file = os.path.join(self.data_dir, "corpus.jsonl")
        print(f"Loading selective corpus from {corpus_file} (需要 {len(needed_doc_ids)} 个文档)...")

        try:
            loaded_count = 0
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading selective corpus"):
                    data = json.loads(line.strip())
                    doc_id = data['_id']
                    if doc_id in needed_doc_ids:
                        self.corpus[doc_id] = {
                            'title': data.get('title', ''),
                            'text': data['text']
                        }
                        loaded_count += 1
                        # 如果已经加载了所有需要的文档，可以提前退出
                        if loaded_count >= len(needed_doc_ids):
                            break
            print(f"Loaded {len(self.corpus)} corpus entries (需要 {len(needed_doc_ids)} 个)")
        except Exception as e:
            print(f"Error loading selective corpus: {e}")
            raise
    
    def _load_qrels(self):
        """加载相关性判断数据"""
        qrels_file = os.path.join(self.data_dir, "qrels", "test.jsonl")
        print(f"Loading qrels from {qrels_file}...")
        
        try:
            with open(qrels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    self.qrels.append({
                        'query-id': data['query-id'],
                        'corpus-id': data['corpus-id'],
                        'score': int(data['score'])
                    })
            
            print(f"Loaded {len(self.qrels)} relevance judgments")
        except Exception as e:
            print(f"Error loading qrels: {e}")
            raise
    
    def _load_bm25_from_file(self) -> Dict[str, Dict]:
        """
        从预下载的 BM25 数据文件中加载数据
        
        支持两种格式：
        1. beir-eval-bm25-top100 格式（JSONL，包含完整文本）
        2. whybe-choi 格式（Arrow，只包含 query-id, corpus-id, score，需要结合原始数据）
        
        Returns:
            字典，格式为 {
                query_id: {
                    'q_text': str,
                    'bm25_results': [{'doc_id': str, 'score': float, 'rank': int, 'text': str, 'title': str}, ...],
                    'qrels': {doc_id: relevance_score, ...}
                }
            }
        """
        if not self.bm25_data_path:
            return {}
        
        # 如果是 whybe-choi 格式（Arrow 文件）
        if self._use_whybe_choi_format and self.bm25_data_path.endswith('.arrow'):
            return self._load_whybe_choi_arrow_file()
        
        # 否则使用 beir-eval-bm25-top100 格式（JSONL）
        print(f"📖 从文件加载 BM25 数据: {self.bm25_data_path}")
        bm25_data = {}
        
        try:
            with open(self.bm25_data_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="加载 BM25 数据"):
                    item = json.loads(line.strip())
                    
                    qid = item.get('qid')
                    if qid is None:
                        qid = item.get('query_id') or item.get('query-id') or item.get('_id')
                    
                    if qid is None:
                        continue
                    
                    query_id = str(qid)
                    q_text = item.get('q_text') or item.get('query') or item.get('text', '')
                    
                    bm25_results = item.get('bm25_results', [])
                    results = []
                    
                    for rank, result in enumerate(bm25_results[:self.bm25_top_k], 1):
                        doc_id = result.get('pid') or result.get('doc_id') or result.get('_id')
                        if not doc_id:
                            continue
                        
                        bm25_score = result.get('bm25_score') or result.get('score', 0.0)
                        text = result.get('text', '')
                        title = result.get('title', '')
                        
                        results.append({
                            'doc_id': str(doc_id),
                            'score': float(bm25_score),
                            'rank': rank,
                            'text': text,
                            'title': title
                        })
                    
                    qrels = item.get('qrels', {})
                    qrels_dict = {}
                    if isinstance(qrels, dict):
                        for doc_id, relevance_score in qrels.items():
                            qrels_dict[str(doc_id)] = int(relevance_score)
                    
                    if results:
                        bm25_data[query_id] = {
                            'q_text': q_text,
                            'bm25_results': results,
                            'qrels': qrels_dict
                        }
            
            print(f"✓ 加载了 {len(bm25_data)} 个查询的 BM25 数据")
            return bm25_data
            
        except Exception as e:
            print(f"❌ 加载 BM25 数据失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _load_whybe_choi_arrow_file(self) -> Dict[str, Dict]:
        """
        从 whybe-choi 格式的 Arrow 文件加载 BM25 粗排数据
        
        Arrow 文件包含：query-id, corpus-id, score
        需要结合原始的 queries 和 corpus 获取文本内容
        
        Returns:
            字典，格式为 {
                query_id: {
                    'q_text': str,
                    'bm25_results': [{'doc_id': str, 'score': float, 'rank': int}, ...],
                    'qrels': {doc_id: relevance_score, ...}
                }
            }
        """
        print(f"📖 从 whybe-choi Arrow 文件加载 BM25 数据: {self.bm25_data_path}")
        
        try:
            # 尝试使用 pyarrow 读取 Arrow 文件
            try:
                import pyarrow as pa
                import pyarrow.ipc as ipc
            except ImportError:
                print("❌ 需要 pyarrow 库来读取 Arrow 文件。请安装: pip install pyarrow")
                return {}
            
            # 读取 Arrow 文件
            with pa.ipc.open_file(self.bm25_data_path, 'r') as reader:
                table = reader.read_all()
                
                # 获取列名
                columns = table.column_names
                print(f"Arrow 文件列名: {columns}")
                
                # 提取数据
                query_ids = []
                corpus_ids = []
                scores = []
                
                if 'query-id' in columns:
                    query_ids = [str(qid.as_py()) for qid in table['query-id']]
                elif 'query_id' in columns:
                    query_ids = [str(qid.as_py()) for qid in table['query_id']]
                
                if 'corpus-id' in columns:
                    corpus_ids = [str(cid.as_py()) for cid in table['corpus-id']]
                elif 'corpus_id' in columns:
                    corpus_ids = [str(cid.as_py()) for cid in table['corpus_id']]
                
                if 'score' in columns:
                    scores = [float(score.as_py()) for score in table['score']]
                
                print(f"✓ 从 Arrow 文件读取了 {len(query_ids)} 条记录")
            
            # 按 query_id 分组，并按 score 排序
            bm25_data = {}
            query_doc_scores = {}  # {query_id: [(doc_id, score), ...]}
            
            for query_id, doc_id, score in zip(query_ids, corpus_ids, scores):
                if query_id not in query_doc_scores:
                    query_doc_scores[query_id] = []
                query_doc_scores[query_id].append((doc_id, score))
            
            # 对每个查询的文档按 score 排序，取 top_k
            for query_id, doc_scores in query_doc_scores.items():
                # 按 score 降序排序
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                # 取 top_k
                top_docs = doc_scores[:self.bm25_top_k]
                
                # 构建 bm25_results
                bm25_results = []
                for rank, (doc_id, score) in enumerate(top_docs, 1):
                    bm25_results.append({
                        'doc_id': str(doc_id),
                        'score': float(score),
                        'rank': rank
                    })
                
                bm25_data[query_id] = {
                    'q_text': '',  # 稍后从 queries 中获取
                    'bm25_results': bm25_results,
                    'qrels': {}  # 稍后从 qrels 中获取
                }
            
            print(f"✓ 处理了 {len(bm25_data)} 个查询的 BM25 数据")
            return bm25_data
            
        except Exception as e:
            print(f"❌ 加载 whybe-choi Arrow 文件失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _process_data(self):
        """处理数据，创建TestQuery对象"""
        print("Processing data...")
        
        if self.use_bm25_retrieval and self.bm25_data_path:
            # 使用预下载的 BM25 数据
            print(f"🔍 使用 BM25 进行粗排（top_k={self.bm25_top_k}）...")
            bm25_data = self._load_bm25_from_file()
            
            if not bm25_data:
                print("⚠️  无法从文件加载 BM25 数据，回退到使用 qrels")
                self.bm25_data_path = None
            else:
                # 构建 qrels 字典以便快速查找相关性分数
                qrels_dict = {}
                for qrel in self.qrels:
                    query_id_key = qrel.get('query-id') or qrel.get('query_id')
                    doc_id_key = qrel.get('corpus-id') or qrel.get('corpus_id') or qrel.get('doc_id')
                    score = qrel.get('score', 0)
                    if query_id_key and doc_id_key:
                        if query_id_key not in qrels_dict:
                            qrels_dict[query_id_key] = {}
                        qrels_dict[query_id_key][doc_id_key] = int(score)
                
                # 使用预下载的 BM25 数据
                for query_id, query_text in tqdm(self.queries.items(), desc="处理 BM25 数据"):
                    # 尝试多种查询 ID 格式匹配
                    matched_data = None
                    if query_id in bm25_data:
                        matched_data = bm25_data[query_id]
                    else:
                        try:
                            query_id_num = int(query_id)
                            if str(query_id_num) in bm25_data:
                                matched_data = bm25_data[str(query_id_num)]
                        except (ValueError, TypeError):
                            pass
                    
                    if not matched_data:
                        print(f"⚠️  警告: 查询 {query_id} 在 BM25 数据中未找到，跳过")
                        continue
                    
                    bm25_results = matched_data.get('bm25_results', [])
                    file_qrels = matched_data.get('qrels', {})
                    
                    query_qrels = qrels_dict.get(query_id, {})
                    query_qrels.update(file_qrels)
                    
                    passages = []
                    valid_passage_ids = []
                    valid_scores = []
                    
                    for bm25_result in bm25_results:
                        doc_id = bm25_result['doc_id']
                        
                        if 'text' in bm25_result and bm25_result['text']:
                            passage_text = bm25_result['text']
                            title = bm25_result.get('title', '')
                            if title:
                                passage_text = f"{title}\n{passage_text}".strip()
                        elif doc_id in self.corpus:
                            passage_text = self.corpus[doc_id]['text']
                        else:
                            print(f"Warning: Passage {doc_id} not found in corpus or BM25 data")
                            continue
                        
                        passages.append(passage_text)
                        valid_passage_ids.append(doc_id)
                        relevance_score = query_qrels.get(doc_id, 0)
                        valid_scores.append(int(relevance_score))
                    
                    if passages:
                        original_positions = list(range(len(passages)))
                        
                        test_query = TestQuery(
                            query_id=query_id,
                            query_text=query_text,
                            passages=passages,
                            passage_ids=valid_passage_ids,
                            relevance_scores=valid_scores,
                            original_positions=original_positions
                        )
                        self.test_queries.append(test_query)
                
                print(f"Processed {len(self.test_queries)} test queries")
                return
        
        # 使用原始的 qrels（默认行为）
        print("📋 使用原始 qrels 数据...")
        
        # 按查询ID分组
        query_groups = {}
        for qrel in self.qrels:
            query_id = qrel['query-id']
            if query_id not in query_groups:
                query_groups[query_id] = {
                    'passage_ids': [],
                    'relevance_scores': []
                }
            
            query_groups[query_id]['passage_ids'].append(qrel['corpus-id'])
            query_groups[query_id]['relevance_scores'].append(qrel['score'])
        
        # 创建TestQuery对象
        for query_id, data in query_groups.items():
            if query_id not in self.queries:
                print(f"Warning: Query {query_id} not found in queries")
                continue
            
            # 获取查询文本
            query_text = self.queries[query_id]
            
            # 获取passage文本
            passages = []
            valid_passage_ids = []
            valid_scores = []
            
            for i, passage_id in enumerate(data['passage_ids']):
                if passage_id in self.corpus:
                    passage_text = self.corpus[passage_id]['text']
                    passages.append(passage_text)
                    valid_passage_ids.append(passage_id)
                    valid_scores.append(data['relevance_scores'][i])
                else:
                    print(f"Warning: Passage {passage_id} not found in corpus")
            
            if passages:  # 只有当有有效passages时才创建TestQuery
                # 创建原始位置
                original_positions = list(range(len(passages)))
                
                test_query = TestQuery(
                    query_id=query_id,
                    query_text=query_text,
                    passages=passages,
                    passage_ids=valid_passage_ids,
                    relevance_scores=valid_scores,
                    original_positions=original_positions
                )
                self.test_queries.append(test_query)
        
        print(f"Processed {len(self.test_queries)} test queries")
    
    def sample_queries(self, queries: List[TestQuery],
                      num_queries: int) -> List[TestQuery]:
        """随机采样查询"""
        if len(queries) <= num_queries:
            return queries

        return random.sample(queries, num_queries)
    
    def get_relevant_passages(self, query: TestQuery) -> List[int]:
        """获取相关passage的索引"""
        relevant_indices = []
        for i, score in enumerate(query.relevance_scores):
            if score > 0:
                relevant_indices.append(i)
        return relevant_indices
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """在父类统计基础上补充分数分布信息。"""
        base_stats = super().get_query_statistics()
        if not base_stats:
            return base_stats

        all_scores = [score for query in self.test_queries for score in query.relevance_scores]
        base_stats.update(
            {
                'relevance_rate': base_stats['relevant_judgments'] / base_stats['total_judgments']
                if base_stats['total_judgments'] else 0,
                'score_distribution': {
                    'min': min(all_scores) if all_scores else 0,
                    'max': max(all_scores) if all_scores else 0,
                    'mean': float(np.mean(all_scores)) if all_scores else 0,
                },
            }
        )
        return base_stats

if __name__ == "__main__":
    # 测试数据加载器
    data_loader = TestDataLoader()
    
    try:
        # 加载数据
        print("\n=== 加载数据 ===")
        test_queries = data_loader.load_all_data()
        
        # 打印统计信息
        stats = data_loader.get_query_statistics()
        print("\n=== 实际数据统计 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # 打印前几个查询的示例
        print("\n=== 查询示例 ===")
        for i, query in enumerate(test_queries[:3]):
            print(f"\n查询 {i+1}:")
            print(f"  ID: {query.query_id}")
            print(f"  文本: {query.query_text}")
            print(f"  Passages数量: {len(query.passages)}")
            print(f"  相关性分数: {query.relevance_scores[:5]}...")  # 只显示前5个
            print(f"  相关passages: {len(data_loader.get_relevant_passages(query))}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()