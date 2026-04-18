"""
TREC DL 2021/2022/2023 数据加载器
支持从本地目录加载 TREC DL 2021/2022/2023 测试集数据
数据格式：
- queries.tsv: query_id \t query_text
- qrels.txt: query_id \t doc_id \t score (或类似格式)
- passage_top100.txt.gz: query_id \t doc_id \t passage_text (BM25 粗排结果)
"""
import gzip
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
class TREC212223Query(BaseQuery):
    """TREC DL 2021/2022/2023 查询数据结构."""


class TREC212223DataLoader(AbstractDataLoader):
    """TREC DL 2021/2022/2023 测试集数据加载器"""

    def __init__(
        self, 
        data_dir: str,
        use_bm25_retrieval: bool = True,  # 默认使用 BM25，因为数据本身就是 BM25 粗排后的
        bm25_top_k: int = 100,
        dataset_name: Optional[str] = None,
    ):
        """
        初始化 TREC DL 2021/2022/2023 数据加载器
        
        Args:
            data_dir: 数据集目录路径（如 "data/trec/trec21"）
            use_bm25_retrieval: 是否使用 BM25 进行粗排（默认 True，因为数据本身就是 BM25 粗排后的）
            bm25_top_k: BM25 检索返回的 top-k 文档数量（默认 100）
            dataset_name: 数据集名称（dl21, dl22, dl23），用于自动识别
        """
        super().__init__(
            data_dir,
            cache_path=None,  # 使用自动生成的缓存路径
            use_bm25_retrieval=use_bm25_retrieval,
            bm25_top_k=bm25_top_k
        )
        
        self.dataset_name = dataset_name
        
        # 尝试从 data_dir 推断数据集名称
        if not self.dataset_name:
            if 'trec21' in data_dir or '2021' in data_dir or 'dl21' in data_dir:
                self.dataset_name = 'dl21'
            elif 'trec22' in data_dir or '2022' in data_dir or 'dl22' in data_dir:
                self.dataset_name = 'dl22'
            elif 'trec23' in data_dir or '2023' in data_dir or 'dl23' in data_dir:
                self.dataset_name = 'dl23'
        
        # 确定文件路径
        self.queries_file = None
        self.qrels_file = None
        self.passage_file = None
        
        # 根据数据集名称确定文件名
        if self.dataset_name == 'dl21':
            self.queries_file = os.path.join(self.data_dir, "2021_queries.tsv")
            self.qrels_file = os.path.join(self.data_dir, "2021.qrels.pass.final.txt")
            self.passage_file = os.path.join(self.data_dir, "2021_passage_top100.txt.gz")
        elif self.dataset_name == 'dl22':
            self.queries_file = os.path.join(self.data_dir, "2022_queries.tsv")
            self.qrels_file = os.path.join(self.data_dir, "2022.qrels.pass.withDupes.txt")
            self.passage_file = os.path.join(self.data_dir, "2022_passage_top100.txt.gz")
        elif self.dataset_name == 'dl23':
            self.queries_file = os.path.join(self.data_dir, "2023_queries.tsv")
            # dl23 可能没有 qrels 文件
            qrels_file_candidates = [
                os.path.join(self.data_dir, "2023.qrels.pass.final.txt"),
                os.path.join(self.data_dir, "2023.qrels.pass.withDupes.txt"),
            ]
            self.qrels_file = None
            for candidate in qrels_file_candidates:
                abs_candidate = os.path.abspath(candidate)
                if os.path.exists(abs_candidate):
                    self.qrels_file = abs_candidate
                    print(f"✓ 找到 qrels 文件: {self.qrels_file}")
                    break
            if not self.qrels_file:
                # 尝试相对路径
                for candidate in qrels_file_candidates:
                    if os.path.exists(candidate):
                        self.qrels_file = os.path.abspath(candidate)
                        print(f"✓ 找到 qrels 文件（相对路径）: {self.qrels_file}")
                        break
            if not self.qrels_file:
                print(f"⚠️  未找到 qrels 文件")
                print(f"   数据目录: {os.path.abspath(self.data_dir)}")
                print(f"   候选路径:")
                for candidate in qrels_file_candidates:
                    abs_candidate = os.path.abspath(candidate)
                    exists = os.path.exists(abs_candidate)
                    print(f"     - {abs_candidate} (存在: {exists})")
            self.passage_file = os.path.join(self.data_dir, "2023_passage_top100.txt.gz")
        else:
            # 尝试自动检测文件
            for year in ['2021', '2022', '2023']:
                queries_candidate = os.path.join(self.data_dir, f"{year}_queries.tsv")
                if os.path.exists(queries_candidate):
                    self.queries_file = queries_candidate
                    # 查找对应的 qrels 和 passage 文件
                    for qrels_pattern in [f"{year}.qrels.pass.final.txt", f"{year}.qrels.pass.withDupes.txt"]:
                        qrels_candidate = os.path.join(self.data_dir, qrels_pattern)
                        if os.path.exists(qrels_candidate):
                            self.qrels_file = qrels_candidate
                            break
                    passage_candidate = os.path.join(self.data_dir, f"{year}_passage_top100.txt.gz")
                    if os.path.exists(passage_candidate):
                        self.passage_file = passage_candidate
                    break
        
        # 验证文件是否存在
        if not self.queries_file or not os.path.exists(self.queries_file):
            raise FileNotFoundError(f"查询文件未找到: {self.queries_file}")
        if not self.passage_file or not os.path.exists(self.passage_file):
            raise FileNotFoundError(f"Passage 文件未找到: {self.passage_file}")
        if self.qrels_file and not os.path.exists(self.qrels_file):
            print(f"⚠️  Qrels 文件未找到: {self.qrels_file}，将不使用相关性标注")
        
        # 查找 msmarco_v2_passage 目录（用于读取实际的 passage 文本）
        # 尝试多个可能的位置
        env_passage_dir = os.getenv("RERANK_MSMARCO_V2_PASSAGE_DIR")
        data_root = Path(os.getenv("RERANK_BIAS_DATA_ROOT", "data")).expanduser()
        possible_passage_dirs = [
            env_passage_dir,
            str(data_root / "msmarco_v2_passage"),
            os.path.join(self.data_dir, "..", "..", "msmarco_v2_passage"),  # data/trec/trec21 -> data/msmarco_v2_passage
            os.path.join(self.data_dir, "..", "msmarco_v2_passage"),  # data/trec/trec21 -> data/trec/msmarco_v2_passage
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.data_dir))), "data", "msmarco_v2_passage"),  # 从项目根目录
        ]
        
        self.passage_dir = None
        for passage_dir in possible_passage_dirs:
            if not passage_dir:
                continue
            abs_passage_dir = os.path.abspath(passage_dir)
            if os.path.exists(abs_passage_dir) and os.path.isdir(abs_passage_dir):
                self.passage_dir = abs_passage_dir
                print(f"✓ 找到 passage 目录: {self.passage_dir}")
                break
        
        if not self.passage_dir:
            print(f"⚠️  警告: 未找到 msmarco_v2_passage 目录，将尝试从其他位置加载 passage 文本")
            print(f"   尝试的路径:")
            for pd in possible_passage_dirs:
                print(f"     - {os.path.abspath(pd)}")

    def load_all_data(self, apply_bm25=None) -> List[TREC212223Query]:
        """
        加载所有数据
        
        Args:
            apply_bm25: 是否应用 BM25 粗排（如果为 None，使用初始化时的 use_bm25_retrieval 设置）
        
        Returns:
            查询列表
        """
        should_apply_bm25 = apply_bm25 if apply_bm25 is not None else self.use_bm25_retrieval
        
        # 检查缓存
        current_cache_path = self._generate_cache_path(should_apply_bm25, self.bm25_top_k)
        if current_cache_path != self.cache_path:
            self.cache_path = current_cache_path
        
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"✅ 检测到缓存文件，从 {self.cache_path} 加载数据...")
            self._load_cache()
            # 检查缓存是否有效（包含查询）
            if len(self.test_queries) == 0:
                print(f"⚠️  缓存文件无效（包含0个查询），删除缓存并重新加载...")
                try:
                    os.remove(self.cache_path)
                except Exception as e:
                    print(f"   删除缓存文件失败: {e}")
            else:
                return self.test_queries
        
        # 加载数据（按照父类的流程）
        self._load_queries()
        self._load_corpus()  # 虽然不加载数据，但为了满足父类流程
        self._load_qrels()
        
        # 更新 use_bm25_retrieval 以便 _process_data 使用
        original_use_bm25 = self.use_bm25_retrieval
        self.use_bm25_retrieval = should_apply_bm25
        
        # 处理数据（调用 _process_data，它会根据 use_bm25_retrieval 选择相应的方法）
        self._process_data()
        
        # 恢复原始设置
        self.use_bm25_retrieval = original_use_bm25
        
        # 保存缓存
        if self.cache_path:
            self._save_cache()
        
        print(f"Successfully loaded {len(self.test_queries)} test queries")
        return self.test_queries
    
    def _has_bm25_applied(self) -> bool:
        """
        检查是否已经在 _process_data 中应用了 BM25
        
        对于 TREC DL 2021/2022/2023，如果 use_bm25_retrieval=True，
        我们会在 _process_data_from_passage_file 中处理 BM25 数据。
        """
        return self.use_bm25_retrieval
    
    def _load_queries(self):
        """加载查询数据"""
        print(f"Loading queries from {self.queries_file}...")
        
        try:
            with open(self.queries_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t', 1)
                    if len(parts) >= 2:
                        query_id = parts[0].strip()
                        query_text = parts[1].strip()
                        self.queries[query_id] = query_text
                    elif len(parts) == 1:
                        # 如果只有一列，可能是格式不同，尝试其他分隔符
                        parts = line.split(' ', 1)
                        if len(parts) >= 2:
                            query_id = parts[0].strip()
                            query_text = parts[1].strip()
                            self.queries[query_id] = query_text
            
            print(f"Loaded {len(self.queries)} queries")
        except Exception as e:
            print(f"Error loading queries: {e}")
            raise
    
    def _load_corpus(self):
        """
        加载语料库数据
        
        注意：对于 TREC DL 2021/2022/2023，passage 文本已经在 passage_top100.txt.gz 文件中，
        不需要单独加载 corpus.jsonl。此方法为了满足抽象基类要求而实现，实际上不加载任何数据。
        """
        # TREC DL 2021/2022/2023 的数据格式不同，passage 文本已经在 passage 文件中
        # 不需要加载单独的 corpus 文件
        print("⚠️  TREC DL 2021/2022/2023 使用 passage 文件，不需要加载单独的 corpus")
        self.corpus = {}  # 初始化为空字典
    
    def _load_passage_text(self, passage_id: str) -> Optional[str]:
        """
        从 msmarco_v2_passage 目录加载 passage 文本
        
        Args:
            passage_id: passage ID，格式如 "msmarco_passage_43_680556381"
        
        Returns:
            passage 文本，如果找不到则返回 None
        """
        if not self.passage_dir:
            return None
        
        # 解析 passage_id: msmarco_passage_43_680556381 -> file_num=43, pid=680556381
        if not passage_id.startswith("msmarco_passage_"):
            return None
        
        parts = passage_id.replace("msmarco_passage_", "").split("_", 1)
        if len(parts) != 2:
            return None
        
        file_num = parts[0]
        pid = parts[1]
        
        # 构建文件路径: msmarco_passage_43
        passage_file = os.path.join(self.passage_dir, f"msmarco_passage_{file_num}")
        
        if not os.path.exists(passage_file):
            return None
        
        # 从文件中读取 passage 文本（JSONL 格式）
        try:
            with open(passage_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("pid") == passage_id:
                            return data.get("passage", "")
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            # 静默失败，返回 None
            pass
        
        return None
    
    def _load_passage_texts_batch(self, passage_ids: List[str]) -> Dict[str, str]:
        """
        批量从 msmarco_v2_passage 目录加载 passage 文本
        
        Args:
            passage_ids: passage ID 列表
        
        Returns:
            {passage_id: passage_text} 字典
        
        Raises:
            FileNotFoundError: 如果找不到文件或某些 passage 不存在
        """
        if not self.passage_dir:
            raise FileNotFoundError(f"passage_dir 未设置，无法加载 passage 文本")
        
        # 按文件分组 passage_ids
        file_groups = {}  # {file_num: [passage_ids]}
        invalid_ids = []  # 记录格式无效的 passage_id
        for passage_id in passage_ids:
            if not passage_id.startswith("msmarco_passage_"):
                invalid_ids.append(passage_id)
                continue
            parts = passage_id.replace("msmarco_passage_", "").split("_", 1)
            if len(parts) != 2:
                invalid_ids.append(passage_id)
                continue
            file_num = parts[0]
            if file_num not in file_groups:
                file_groups[file_num] = []
            file_groups[file_num].append(passage_id)
        
        if invalid_ids:
            raise ValueError(f"发现 {len(invalid_ids)} 个格式无效的 passage_id（前10个）: {invalid_ids[:10]}")
        
        # 从每个文件加载 passages
        passage_texts = {}
        missing_files = []  # 记录找不到的文件
        missing_passages = {}  # {file_num: [missing_passage_ids]}
        
        for file_num, pids in tqdm(file_groups.items(), desc="加载 Passage 文本"):
            passage_file = os.path.join(self.passage_dir, f"msmarco_passage_{file_num}")
            if not os.path.exists(passage_file):
                missing_files.append(passage_file)
                missing_passages[file_num] = pids
                continue
            
            # 创建 pid 集合以便快速查找
            pid_set = set(pids)
            
            try:
                with open(passage_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            pid = data.get("pid")
                            if pid in pid_set:
                                passage_text = data.get("passage", "")
                                if not passage_text or not passage_text.strip():
                                    # passage 文本为空，记录但继续
                                    if file_num not in missing_passages:
                                        missing_passages[file_num] = []
                                    missing_passages[file_num].append(pid)
                                else:
                                    passage_texts[pid] = passage_text
                                pid_set.remove(pid)
                                # 如果所有 passages 都已找到，可以提前退出
                                if not pid_set:
                                    break
                        except json.JSONDecodeError:
                            continue
                
                # 检查是否还有未找到的 passages
                if pid_set:
                    if file_num not in missing_passages:
                        missing_passages[file_num] = []
                    missing_passages[file_num].extend(list(pid_set))
            except Exception as e:
                raise RuntimeError(f"读取文件 {passage_file} 时出错: {e}")
        
        # 检查是否有缺失的文件或 passages
        if missing_files:
            raise FileNotFoundError(
                f"找不到以下 passage 文件（共 {len(missing_files)} 个）:\n" +
                "\n".join(f"  - {f}" for f in missing_files[:10]) +
                (f"\n  ... 还有 {len(missing_files) - 10} 个文件" if len(missing_files) > 10 else "")
            )
        
        if missing_passages:
            total_missing = sum(len(pids) for pids in missing_passages.values())
            missing_samples = []
            for file_num, pids in list(missing_passages.items())[:5]:
                missing_samples.extend(pids[:3])
            
            raise ValueError(
                f"在 msmarco_v2_passage 目录中找不到 {total_missing} 个 passage 文本！\n"
                f"这不应该发生，因为所有文章都应该在该目录中。\n"
                f"缺失的 passage 示例（前15个）: {missing_samples}\n"
                f"请检查:\n"
                f"1. passage_dir 路径是否正确: {self.passage_dir}\n"
                f"2. passage_id 格式是否正确\n"
                f"3. 数据文件是否完整"
            )
        
        return passage_texts
    
    def _load_qrels(self):
        """加载相关性判断数据"""
        if not self.qrels_file or not os.path.exists(self.qrels_file):
            print("⚠️  Qrels 文件不存在，跳过加载")
            return
        
        print(f"Loading qrels from {self.qrels_file}...")
        
        try:
            with open(self.qrels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        query_id = parts[0].strip()
                        doc_id = parts[2].strip()  # 格式: query_id iteration doc_id score
                        score = int(parts[3]) if len(parts) >= 4 else 1
                        self.qrels.append({
                            'query-id': query_id,
                            'corpus-id': doc_id,
                            'score': score
                        })
                    elif len(parts) == 3:
                        # 可能是 query_id \t doc_id \t score 格式
                        query_id = parts[0].strip()
                        doc_id = parts[1].strip()
                        score = int(parts[2])
                        self.qrels.append({
                            'query-id': query_id,
                            'corpus-id': doc_id,
                            'score': score
                        })
            
            print(f"Loaded {len(self.qrels)} relevance judgments")
        except Exception as e:
            print(f"Error loading qrels: {e}")
            # 不抛出异常，因为某些数据集可能没有 qrels
    
    def _process_data_from_passage_file(self):
        """从 passage_top100.txt.gz 文件处理数据（BM25 粗排后的结果）"""
        print(f"🔍 从 Passage 文件处理数据（top_k={self.bm25_top_k}）...")
        
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
        
        # 从文件加载 passage 数据（TREC 格式：query_id Q0 passage_id rank score Anserini）
        query_passages = {}  # {query_id: [(doc_id, rank), ...]}
        all_passage_ids = set()  # 收集所有 passage_id 以便批量加载
        
        try:
            # 支持压缩和未压缩的文件
            if self.passage_file.endswith('.gz'):
                # 如果 .gz 文件不存在，尝试未压缩的文件
                if not os.path.exists(self.passage_file):
                    uncompressed_file = self.passage_file[:-3]  # 移除 .gz 后缀
                    if os.path.exists(uncompressed_file):
                        print(f"⚠️  .gz 文件不存在，使用未压缩文件: {uncompressed_file}")
                        self.passage_file = uncompressed_file
                file_opener = gzip.open(self.passage_file, 'rt', encoding='utf-8')
            else:
                file_opener = open(self.passage_file, 'r', encoding='utf-8')
            
            with file_opener as f:
                line_count = 0
                sample_lines = []  # 保存前几行用于调试
                for line in tqdm(f, desc="加载 Passage 排名数据"):
                    line = line.strip()
                    if not line:
                        continue
                    
                    line_count += 1
                    # 保存前5行用于调试
                    if line_count <= 5:
                        sample_lines.append(line)
                    
                    # 解析 TREC 格式: query_id Q0 passage_id rank score Anserini
                    parts = line.split()
                    if len(parts) >= 3:
                        query_id = parts[0].strip()
                        doc_id = parts[2].strip()  # passage_id 在第3个位置（索引2）
                        rank = int(parts[3]) if len(parts) >= 4 else len(query_passages.get(query_id, [])) + 1
                        
                        if query_id and doc_id:
                            if query_id not in query_passages:
                                query_passages[query_id] = []
                            query_passages[query_id].append((doc_id, rank))
                            all_passage_ids.add(doc_id)
                
                # 如果解析失败，打印样本行用于调试
                if len(query_passages) == 0 and sample_lines:
                    print(f"⚠️  警告: 未能解析任何 Passage 数据")
                    print(f"   文件总行数: {line_count}")
                    print(f"   前5行样本:")
                    for i, sample in enumerate(sample_lines[:5], 1):
                        print(f"     行 {i}: {sample[:200]}...")  # 只显示前200个字符
            
            print(f"✓ 加载了 {len(query_passages)} 个查询的 Passage 排名数据")
            if query_passages:
                total_passages = sum(len(passages) for passages in query_passages.values())
                print(f"   总共 {total_passages} 个 passages")
        except Exception as e:
            print(f"❌ 加载 Passage 数据失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 批量加载 passage 文本
        if not self.passage_dir:
            raise FileNotFoundError(
                f"未找到 msmarco_v2_passage 目录，无法加载 passage 文本。\n"
                f"请确保 msmarco_v2_passage 目录存在于以下位置之一:\n" +
                "\n".join(f"  - {os.path.abspath(pd)}" for pd in possible_passage_dirs)
            )
        
        print(f"📖 从 msmarco_v2_passage 目录加载 {len(all_passage_ids)} 个 passage 文本...")
        # 如果找不到任何 passage，_load_passage_texts_batch 会抛出错误
        passage_texts = self._load_passage_texts_batch(list(all_passage_ids))
        print(f"✓ 成功加载 {len(passage_texts)} 个 passage 文本")
        
        # 验证所有 passage 都已加载
        if len(passage_texts) < len(all_passage_ids):
            missing_ids = set(all_passage_ids) - set(passage_texts.keys())
            raise ValueError(
                f"未能加载所有 passage 文本！缺失 {len(missing_ids)} 个 passage。\n"
                f"缺失的 passage 示例（前10个）: {list(missing_ids)[:10]}\n"
                f"这不应该发生，因为所有文章都应该在 msmarco_v2_passage 目录中。"
            )
        
        # 处理数据，创建 TestQuery 对象
        processed_count = 0
        skipped_count = 0
        all_zero_queries_count = 0  # 统计所有passages相关性分数都为0的查询数
        
        for query_id, query_text in tqdm(self.queries.items(), desc="处理查询数据"):
            if query_id not in query_passages:
                skipped_count += 1
                continue
            
            # 获取该查询的 passages（已按 BM25 排序）
            # passages_data 格式: [(doc_id, rank), ...]
            passages_data = sorted(query_passages[query_id], key=lambda x: x[1])[:self.bm25_top_k]
            
            # 去重：如果同一个passage_id出现多次，只保留rank最小的那个
            seen_doc_ids = set()
            deduplicated_passages = []
            for doc_id, rank in passages_data:
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    deduplicated_passages.append((doc_id, rank))
            passages_data = deduplicated_passages
            
            # 获取该查询的 qrels
            query_qrels = qrels_dict.get(query_id, {})
            
            # 先检查是否所有passages的相关性分数都是0
            all_scores = [query_qrels.get(doc_id, 0) for doc_id, rank in passages_data]
            if all(score <= 0 for score in all_scores):
                # 如果所有passages的相关性分数都是0，跳过这个查询
                all_zero_queries_count += 1
                continue
            
            # 构建 passages 和相关信息（保留所有文档，包括分数为0的）
            passages = []
            valid_passage_ids = []
            valid_scores = []
            
            for doc_id, rank in passages_data:
                # 尝试从 qrels 中获取真实的相关性分数，如果不存在则设为 0
                relevance_score = query_qrels.get(doc_id, 0)
                
                # 从 passage_texts 字典中获取 passage 文本
                # 由于 _load_passage_texts_batch 已经确保所有 passage 都存在，这里应该总是能找到
                passage_text = passage_texts.get(doc_id)
                if not passage_text:
                    raise ValueError(
                        f"查询 {query_id} 的 passage {doc_id} 未在 passage_texts 中找到！\n"
                        f"这不应该发生，因为所有文章都应该在 msmarco_v2_passage 目录中。"
                    )
                
                # 确保 passage_text 不是空字符串
                passage_text = passage_text.strip()
                if not passage_text:
                    raise ValueError(
                        f"查询 {query_id} 的 passage {doc_id} 的文本为空！\n"
                        f"这不应该发生，请检查数据文件。"
                    )
                
                passages.append(passage_text)
                valid_passage_ids.append(doc_id)
                valid_scores.append(int(relevance_score))
            
            if passages:  # 只有当有有效passages时才创建TestQuery
                original_positions = list(range(len(passages)))
                
                test_query = TREC212223Query(
                    query_id=query_id,
                    query_text=query_text,
                    passages=passages,
                    passage_ids=valid_passage_ids,
                    relevance_scores=valid_scores,
                    original_positions=original_positions
                )
                self.test_queries.append(test_query)
                processed_count += 1
        
        print(f"Processed {processed_count} test queries from Passage file")
        if skipped_count > 0:
            print(f"⚠️  跳过了 {skipped_count} 个没有 Passage 数据的查询")
        if all_zero_queries_count > 0:
            print(f"✓ 过滤掉了 {all_zero_queries_count} 个所有passages相关性分数都为0的查询")
    
    def _process_data_from_qrels(self):
        """从 qrels 处理数据（不使用 BM25 粗排）"""
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
        
        # 收集所有需要的 passage_ids
        all_passage_ids = set()
        for query_id, data in query_groups.items():
            all_passage_ids.update(data['passage_ids'])
        
        # 从 msmarco_v2_passage 目录加载所有 passage 文本
        # 如果找不到任何 passage，_load_passage_texts_batch 会抛出错误
        print(f"📖 从 msmarco_v2_passage 目录加载 {len(all_passage_ids)} 个 passage 文本...")
        doc_texts = self._load_passage_texts_batch(list(all_passage_ids))
        print(f"✓ 成功加载 {len(doc_texts)} 个 passage 文本")
        
        # 验证所有 passage 都已加载
        if len(doc_texts) < len(all_passage_ids):
            missing_ids = set(all_passage_ids) - set(doc_texts.keys())
            raise ValueError(
                f"未能加载所有 passage 文本！缺失 {len(missing_ids)} 个 passage。\n"
                f"缺失的 passage 示例（前10个）: {list(missing_ids)[:10]}\n"
                f"这不应该发生，因为所有文章都应该在 msmarco_v2_passage 目录中。"
            )
        
        # 创建TestQuery对象
        all_zero_queries_count = 0  # 统计所有passages相关性分数都为0的查询数
        
        for query_id, data in query_groups.items():
            if query_id not in self.queries:
                print(f"Warning: Query {query_id} not found in queries")
                continue
            
            # 先检查是否所有passages的相关性分数都是0
            all_scores = data['relevance_scores']
            if all(score <= 0 for score in all_scores):
                # 如果所有passages的相关性分数都是0，跳过这个查询
                all_zero_queries_count += 1
                continue
            
            # 获取查询文本
            query_text = self.queries[query_id]
            
            # 获取passage文本（保留所有文档，包括分数为0的）
            passages = []
            valid_passage_ids = []
            valid_scores = []
            
            for i, passage_id in enumerate(data['passage_ids']):
                # 获取相关性分数
                relevance_score = data['relevance_scores'][i]
                
                # 由于 _load_passage_texts_batch 已经确保所有 passage 都存在，这里应该总是能找到
                passage_text = doc_texts.get(passage_id)
                if not passage_text:
                    raise ValueError(
                        f"查询 {query_id} 的 passage {passage_id} 未在 doc_texts 中找到！\n"
                        f"这不应该发生，因为所有文章都应该在 msmarco_v2_passage 目录中。"
                    )
                
                # 确保 passage_text 不是空字符串
                passage_text = passage_text.strip()
                if not passage_text:
                    raise ValueError(
                        f"查询 {query_id} 的 passage {passage_id} 的文本为空！\n"
                        f"这不应该发生，请检查数据文件。"
                    )
                
                passages.append(passage_text)
                valid_passage_ids.append(passage_id)
                valid_scores.append(relevance_score)
            
            if passages:  # 只有当有有效passages时才创建TestQuery
                # 创建原始位置
                original_positions = list(range(len(passages)))
                
                test_query = TREC212223Query(
                    query_id=query_id,
                    query_text=query_text,
                    passages=passages,
                    passage_ids=valid_passage_ids,
                    relevance_scores=valid_scores,
                    original_positions=original_positions
                )
                self.test_queries.append(test_query)
        
        print(f"Processed {len(self.test_queries)} test queries")
        if all_zero_queries_count > 0:
            print(f"✓ 过滤掉了 {all_zero_queries_count} 个所有passages相关性分数都为0的查询")
    
    def _process_data(self):
        """
        处理数据，创建 TestQuery 对象
        
        这是抽象基类要求的抽象方法。根据 use_bm25_retrieval 设置，
        调用相应的处理方法。
        """
        if self.use_bm25_retrieval:
            self._process_data_from_passage_file()
        else:
            # 如果没有 qrels 文件，回退到使用 passage 文件
            if not self.qrels_file or not os.path.exists(self.qrels_file) or len(self.qrels) == 0:
                print("⚠️  没有 qrels 数据，回退到使用 passage 文件...")
                self._process_data_from_passage_file()
            else:
                self._process_data_from_qrels()
    
    def sample_queries(self, queries: List[TREC212223Query],
                      num_queries: int) -> List[TREC212223Query]:
        """随机采样查询"""
        if len(queries) <= num_queries:
            return queries

        return random.sample(queries, num_queries)
    
    def get_relevant_passages(self, query: TREC212223Query) -> List[int]:
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
    import sys
    from pathlib import Path
    
    # 自动检测数据目录路径
    # 尝试从当前文件位置推断项目根目录
    current_file = Path(__file__).resolve()
    # 当前文件在: code/dataloader/dl_21_22_23_loader.py
    # 项目根目录应该是: code 的父目录
    code_dir = current_file.parent.parent  # code 目录
    project_root = code_dir.parent  # 项目根目录
    
    # 尝试多个可能的数据目录路径
    possible_data_dirs = [
        project_root / "data" / "trec" / "trec21",  # 从项目根目录
        Path("data") / "trec" / "trec21",  # 相对路径（从当前工作目录）
        Path("../data") / "trec" / "trec21",  # 从 code 目录
    ]
    
    data_dir = None
    for possible_dir in possible_data_dirs:
        if possible_dir.exists() and (possible_dir / "2021_queries.tsv").exists():
            data_dir = str(possible_dir)
            break
    
    if not data_dir:
        # 如果都找不到，使用默认路径
        data_dir = str(project_root / "data" / "trec" / "trec21")
        print(f"⚠️  未找到数据目录，将尝试: {data_dir}")
    
    # 测试 dl21
    print("\n=== 测试 dl21 ===")
    print(f"数据目录: {data_dir}")
    try:
        data_loader = TREC212223DataLoader(
            data_dir=data_dir,
            dataset_name="dl21"
        )
        test_queries = data_loader.load_all_data()
        print(f"✓ 成功加载 {len(test_queries)} 个查询")
        
        if test_queries:
            stats = data_loader.get_query_statistics()
            print("\n=== 统计信息 ===")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")
            
            # 显示第一个查询的示例
            print("\n=== 查询示例（第一个查询）===")
            first_query = test_queries[0]
            print(f"查询 ID: {first_query.query_id}")
            print(f"查询文本: {first_query.query_text[:150]}...")
            print(f"Passages 数量: {len(first_query.passages)}")
            print(f"相关性分数（前10个）: {first_query.relevance_scores[:10]}")
            relevant_count = sum(1 for s in first_query.relevance_scores if s > 0)
            print(f"相关 passages 数量: {relevant_count}")
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print(f"   请确保数据文件存在于: {data_dir}")
        print(f"   尝试的路径:")
        for pd in possible_data_dirs:
            print(f"     - {pd} (存在: {pd.exists()})")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

