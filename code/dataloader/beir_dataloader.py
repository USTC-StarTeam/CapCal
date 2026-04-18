"""
BEIR loader for the cleaned paper release.

The public repository keeps only the `trec-covid` path that is used in the
paper's main experiments.
"""

import os
import json
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False
    print("Warning: rank_bm25 not available. Install with: pip install rank-bm25")

try:
    from beir.datasets.data_loader import GenericDataLoader
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False
    print("Warning: beir library not available. Install with: pip install beir")

try:
    from ..src.abstract_dl import AbstractDataLoader, BaseQuery
except ImportError:
    import sys
    CURRENT_DIR = Path(__file__).resolve().parent
    CODE_DIR = CURRENT_DIR.parent
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    from src.abstract_dl import AbstractDataLoader, BaseQuery


class BEIRDataLoader(AbstractDataLoader):
    """BEIR dataset loader."""

    BEIR_DATASETS = ["trec-covid"]
    
    def __init__(
        self, 
        data_dir: str, 
        dataset_name: str,
        use_bm25_retrieval: bool = False,
        bm25_top_k: int = 100,
        bm25_data_path: Optional[str] = None,
    ):
        """
        初始化 BEIR 数据加载器
        
        Args:
            data_dir: 数据集目录路径（完整路径，如 "data/beir/trec-covid"）
            dataset_name: 数据集名称（如 'trec-covid'）
            use_bm25_retrieval: 是否使用 BM25 进行粗排（如果为 True，将使用 BM25 检索 top-k 文档，而不是使用 qrels）
            bm25_top_k: BM25 检索返回的 top-k 文档数量（默认 100）
            bm25_data_path: 预下载的 BM25 数据路径（可选，如果提供则从文件读取，否则实时计算）
                          可以是文件路径或目录路径（如果是目录，会自动查找 bm25_top100.jsonl）
        """
        self.dataset_name = dataset_name
        if dataset_name not in self.BEIR_DATASETS:
            raise ValueError(f"Unknown BEIR dataset: {dataset_name}. Available: {self.BEIR_DATASETS}")
        
        # 传递参数给父类（包括 BM25 参数）
        super().__init__(
            data_dir, 
            cache_path=None,  # 使用自动生成的缓存路径
            use_bm25_retrieval=use_bm25_retrieval, 
            bm25_top_k=bm25_top_k
        )
        
        self.split = "test"
        
        # 标记：子类已经在 _process_data 中处理了 BM25（如果启用）
        self._bm25_handled_in_process = use_bm25_retrieval
        
        # BM25 索引（延迟初始化，仅在需要实时计算时使用）
        self._bm25_index = None
        self._corpus_texts = []
        self._corpus_tokenized = []
        self._corpus_ids_list = []
        
        # 预下载的 BM25 数据路径
        self.bm25_data_path = bm25_data_path
        if self.bm25_data_path:
            # 如果是目录，查找 bm25_top100.jsonl 文件
            bm25_path = Path(self.bm25_data_path)
            if bm25_path.is_dir():
                bm25_file = bm25_path / "bm25_top100.jsonl"
                if bm25_file.exists():
                    self.bm25_data_path = str(bm25_file)
                else:
                    # 尝试查找其他可能的文件名
                    jsonl_files = list(bm25_path.glob("*.jsonl"))
                    if jsonl_files:
                        self.bm25_data_path = str(jsonl_files[0])
                        print(f"📁 找到 BM25 数据文件: {self.bm25_data_path}")
                    else:
                        print(f"⚠️  警告: 在目录 {bm25_path} 中未找到 BM25 数据文件，将使用实时计算")
                        self.bm25_data_path = None
            elif not bm25_path.exists():
                print(f"⚠️  警告: BM25 数据文件不存在: {self.bm25_data_path}，将使用实时计算")
                self.bm25_data_path = None
        else:
            # 如果没有指定路径，尝试自动查找
            # 实际文件位置: data/beir-eval-bm25-top100/{dataset_name}.jsonl
            # 文件名格式：trec-covid -> trec_covid.jsonl, nfcorpus -> nfcorpus.jsonl
            try:
                # 尝试获取 CODE_DIR（可能在 try-except 块中定义）
                current_dir = Path(__file__).resolve().parent
                code_dir = current_dir.parent
                base_data_dir = code_dir.parent / "data"
            except:
                # 如果无法获取，使用相对路径
                base_data_dir = Path("data")
            
            # 转换数据集名称格式：trec-covid -> trec_covid
            dataset_file_name = dataset_name.replace('-', '_')
            # 尝试多个可能的文件名
            possible_files = [
                base_data_dir / "beir-eval-bm25-top100" / f"{dataset_file_name}.jsonl",
                base_data_dir / "beir-eval-bm25-top100" / f"{dataset_name}.jsonl",
                base_data_dir / "beir-bm25-top100" / dataset_name / "bm25_top100.jsonl",
            ]
            
            for bm25_file in possible_files:
                if bm25_file.exists():
                    self.bm25_data_path = str(bm25_file)
                    print(f"📁 自动找到 BM25 数据文件: {self.bm25_data_path}")
                    break
    
    def _load_queries(self):
        """加载查询数据"""
        if not BEIR_AVAILABLE:
            raise ImportError("beir library is required. Install with: pip install beir")
        
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
                    text = data.get('text', '')
                    title = data.get('title', '')
                    merged = f"{title}\n{text}".strip()
                    self.corpus[data['_id']] = {
                        'title': title,
                        'text': merged
                    }
            
            print(f"Loaded {len(self.corpus)} corpus entries")
        except Exception as e:
            print(f"Error loading corpus: {e}")
            raise
    
    def _load_qrels(self):
        """加载相关性判断数据（BEIR 使用 TSV 格式）"""
        qrels_file = os.path.join(self.data_dir, "qrels", f"{self.split}.tsv")
        print(f"Loading qrels from {qrels_file}...")
        
        try:
            # BEIR 使用 TSV 格式：query-id corpus-id score
            qrels_dict = {}
            with open(qrels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    header_token = parts[0].lower()
                    if header_token in {'query-id', 'query_id'}:
                        continue
                    try:
                        score_val = int(parts[2])
                    except ValueError:
                        print(f"⚠️ 跳过无法解析的 qrels 行: {line.strip()}")
                        continue
                    qid, doc_id = parts[0], parts[1]
                    qrels_dict.setdefault(qid, {})[doc_id] = score_val
            
            # 转换为父类期望的格式
            for qid, docs in qrels_dict.items():
                for doc_id, score in docs.items():
                    self.qrels.append({
                        'query-id': qid,
                        'corpus-id': doc_id,
                        'score': score
                    })
            
            print(f"Loaded {len(self.qrels)} relevance judgments")
        except Exception as e:
            print(f"Error loading qrels: {e}")
            raise
    
    def _build_bm25_index(self):
        """构建 BM25 索引"""
        if not RANK_BM25_AVAILABLE:
            raise ImportError("rank_bm25 library is required for BM25 retrieval. Install with: pip install rank-bm25")
        
        if self._bm25_index is not None:
            return
        
        print(f"🔨 构建 BM25 索引（语料库大小: {len(self.corpus)}）...")
        
        # 准备语料库文本和 tokenized 版本
        self._corpus_texts = []
        self._corpus_tokenized = []
        self._corpus_ids_list = []
        
        for doc_id, doc_data in tqdm(self.corpus.items(), desc="Tokenizing corpus"):
            text = doc_data['text']
            # 简单的 tokenization（按空格和标点分割）
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
    
    def _load_bm25_from_file(self) -> Dict[str, Dict]:
        """
        从预下载的 BM25 数据文件中加载数据
        
        实际文件格式：
        {
            "qid": 44,
            "q_text": "查询文本",
            "bm25_results": [
                {"text": "...", "title": "...", "bm25_score": 9.45, "pid": "doc_id"},
                ...
            ],
            "qrels": {"doc_id": relevance_score, ...}
        }
        
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
        
        print(f"📖 从文件加载 BM25 数据: {self.bm25_data_path}")
        bm25_data = {}
        
        try:
            with open(self.bm25_data_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="加载 BM25 数据"):
                    item = json.loads(line.strip())
                    
                    # 实际格式：qid (可能是数字或字符串), q_text, bm25_results, qrels
                    qid = item.get('qid')
                    if qid is None:
                        # 尝试其他可能的键名
                        qid = item.get('query_id') or item.get('query-id') or item.get('_id')
                    
                    if qid is None:
                        continue
                    
                    # 转换为字符串以匹配 queries 中的格式
                    query_id = str(qid)
                    q_text = item.get('q_text') or item.get('query') or item.get('text', '')
                    
                    # 解析 bm25_results
                    bm25_results = item.get('bm25_results', [])
                    results = []
                    
                    for rank, result in enumerate(bm25_results[:self.bm25_top_k], 1):
                        # 实际格式中，doc_id 在 'pid' 字段中
                        doc_id = result.get('pid') or result.get('doc_id') or result.get('_id')
                        if not doc_id:
                            continue
                        
                        # bm25_score 在 'bm25_score' 字段中
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
                    
                    # 解析 qrels（如果存在）
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
    
    def load_all_data(self, apply_bm25=None):
        """
        重写父类方法，如果使用预下载的 BM25 文件，跳过加载 corpus
        
        Args:
            apply_bm25: 是否应用 BM25 粗排（如果为 None，使用初始化时的 use_bm25_retrieval 设置）
        
        Returns:
            查询列表
        """
        should_apply_bm25 = apply_bm25 if apply_bm25 is not None else self.use_bm25_retrieval
        
        # 如果使用 BM25 且有预下载的文件，优化加载流程
        if should_apply_bm25 and self.bm25_data_path:
            print(f"📖 检测到预下载的 BM25 文件，直接使用（跳过加载完整 corpus）...")
            
            # 只加载查询（用于匹配）和 qrels（用于验证），不加载 corpus
            self._load_queries()
            self._load_qrels()
            
            # 直接从 BM25 文件处理数据
            self._process_data_from_bm25_file()
            
            # 保存缓存
            current_cache_path = self._generate_cache_path(should_apply_bm25, self.bm25_top_k)
            if current_cache_path != self.cache_path:
                self.cache_path = current_cache_path
            
            if self.cache_path:
                self._save_cache()
            
            return self.test_queries
        
        # 否则使用父类的标准流程
        return super().load_all_data(apply_bm25)
    
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
        
        # 从文件加载 BM25 数据
        bm25_data = self._load_bm25_from_file()
        
        if not bm25_data:
            raise RuntimeError("无法从 BM25 文件加载数据")
        
        # 直接从 BM25 文件处理数据，而不是从 queries.jsonl 遍历
        # 这样可以确保处理所有在 BM25 文件中的查询
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
                
                # 直接从 BM25 文件获取文本
                if 'text' in bm25_result and bm25_result['text']:
                    passage_text = bm25_result['text']
                    title = bm25_result.get('title', '')
                    if title:
                        passage_text = f"{title}\n{passage_text}".strip()
                else:
                    # 如果 BM25 文件中没有文本，尝试从 corpus 获取（但通常不应该发生）
                    if doc_id in self.corpus:
                        passage_text = self.corpus[doc_id]['text']
                    else:
                        continue  # 跳过没有文本的 passage
                
                passages.append(passage_text)
                valid_passage_ids.append(doc_id)
                # 尝试从 qrels 中获取真实的相关性分数，如果不存在则设为 0
                relevance_score = query_qrels.get(doc_id, 0)
                valid_scores.append(int(relevance_score))
            
            if passages:  # 只有当有有效passages时才创建BaseQuery
                # 创建原始位置
                original_positions = list(range(len(passages)))
                
                test_query = BaseQuery(
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
    
    def _process_data(self):
        """处理数据，创建BaseQuery对象"""
        print("Processing data...")
        
        if self.use_bm25_retrieval:
            print(f"🔍 使用 BM25 进行粗排（top_k={self.bm25_top_k}）...")
            
            # 构建 qrels 字典以便快速查找相关性分数
            qrels_dict = {}
            for qrel in self.qrels:
                query_id_key = qrel.get('query-id') or qrel.get('query_id')
                doc_id_key = qrel.get('corpus-id') or qrel.get('corpus_id') or qrel.get('doc_id')
                score = qrel.get('score', 0)
                if query_id_key and doc_id_key:
                    if query_id_key not in qrels_dict:
                        qrels_dict[query_id_key] = {}
                    qrels_dict[query_id_key][doc_id_key] = int(score)  # 确保是整数
            
            # 检查是否使用预下载的 BM25 数据
            if self.bm25_data_path:
                # 从文件加载 BM25 数据
                bm25_data = self._load_bm25_from_file()
                
                if not bm25_data:
                    print("⚠️  无法从文件加载 BM25 数据，回退到实时计算")
                    self.bm25_data_path = None  # 清除路径，使用实时计算
                else:
                    # 使用预下载的 BM25 数据
                    for query_id, query_text in tqdm(self.queries.items(), desc="处理 BM25 数据"):
                        # 尝试多种查询 ID 格式匹配
                        matched_data = None
                        if query_id in bm25_data:
                            matched_data = bm25_data[query_id]
                        else:
                            # 尝试数字格式匹配
                            try:
                                query_id_num = int(query_id)
                                if str(query_id_num) in bm25_data:
                                    matched_data = bm25_data[str(query_id_num)]
                            except (ValueError, TypeError):
                                pass
                        
                        if not matched_data:
                            print(f"⚠️  警告: 查询 {query_id} 在 BM25 数据中未找到，跳过")
                            continue
                        
                        # 获取该查询的 BM25 结果和 qrels
                        bm25_results = matched_data.get('bm25_results', [])
                        file_qrels = matched_data.get('qrels', {})
                        
                        # 合并文件中的 qrels 和原始的 qrels_dict
                        query_qrels = qrels_dict.get(query_id, {})
                        # 文件中的 qrels 优先级更高
                        query_qrels.update(file_qrels)
                        
                        # 获取passage文本
                        passages = []
                        valid_passage_ids = []
                        valid_scores = []
                        
                        for bm25_result in bm25_results:
                            doc_id = bm25_result['doc_id']
                            
                            # 优先使用文件中的文本，否则从 corpus 中获取
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
                            # 尝试从 qrels 中获取真实的相关性分数，如果不存在则设为 0
                            relevance_score = query_qrels.get(doc_id, 0)
                            valid_scores.append(int(relevance_score))  # 确保是整数
                        
                        if passages:  # 只有当有有效passages时才创建BaseQuery
                            # 创建原始位置
                            original_positions = list(range(len(passages)))
                            
                            test_query = BaseQuery(
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
            
            # 如果没有预下载数据或加载失败，使用实时计算
            if not RANK_BM25_AVAILABLE:
                raise ImportError("rank_bm25 library is required. Install with: pip install rank-bm25")
            
            # 使用 BM25 检索
            for query_id, query_text in tqdm(self.queries.items(), desc="BM25 Retrieval"):
                # 使用 BM25 检索 top-k 文档
                bm25_results = self._retrieve_with_bm25(query_text, self.bm25_top_k)
                
                # 获取该查询的 qrels（如果存在）
                query_qrels = qrels_dict.get(query_id, {})
                
                # 获取passage文本
                passages = []
                valid_passage_ids = []
                valid_scores = []
                
                for doc_id, bm25_score, rank in bm25_results:
                    if doc_id in self.corpus:
                        passage_text = self.corpus[doc_id]['text']
                        passages.append(passage_text)
                        valid_passage_ids.append(doc_id)
                        # 尝试从 qrels 中获取真实的相关性分数，如果不存在则设为 0
                        relevance_score = query_qrels.get(doc_id, 0)
                        valid_scores.append(int(relevance_score))  # 确保是整数
                    else:
                        print(f"Warning: Passage {doc_id} not found in corpus")
                
                if passages:  # 只有当有有效passages时才创建BaseQuery
                    # 创建原始位置
                    original_positions = list(range(len(passages)))
                    
                    test_query = BaseQuery(
                        query_id=query_id,
                        query_text=query_text,
                        passages=passages,
                        passage_ids=valid_passage_ids,
                        relevance_scores=valid_scores,
                        original_positions=original_positions
                    )
                    self.test_queries.append(test_query)
        else:
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
            
            # 创建BaseQuery对象
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
                
                if passages:  # 只有当有有效passages时才创建BaseQuery
                    # 创建原始位置
                    original_positions = list(range(len(passages)))
                    
                    test_query = BaseQuery(
                        query_id=query_id,
                        query_text=query_text,
                        passages=passages,
                        passage_ids=valid_passage_ids,
                        relevance_scores=valid_scores,
                        original_positions=original_positions
                    )
                    self.test_queries.append(test_query)
        
        print(f"Processed {len(self.test_queries)} test queries")
    
    @classmethod
    def list_datasets(cls) -> list:
        """列出所有可用的 BEIR 数据集"""
        return cls.BEIR_DATASETS.copy()
