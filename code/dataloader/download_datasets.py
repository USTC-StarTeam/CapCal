"""Download helpers for the cleaned TREC-COVID / TREC-DL reproduction setup."""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

# 添加项目路径
CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader.factory import DataLoaderFactory


DEFAULT_DATA_ROOT = Path(os.getenv("RERANK_BIAS_DATA_ROOT", "data")).expanduser()

try:
    from beir import util
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False
    print("Warning: beir library not available. Install with: pip install beir")


class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, base_data_dir: str = str(DEFAULT_DATA_ROOT)):
        """
        初始化下载器
        
        Args:
            base_data_dir: 基础数据目录
        """
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        
        # BEIR 数据集下载 URL 模板
        self.BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
        
        self.supported_datasets = DataLoaderFactory.list_supported_datasets()
    
    def download_beir_dataset(self, dataset_name: str, force_redownload: bool = False) -> Optional[str]:
        """
        下载 BEIR 数据集
        
        Args:
            dataset_name: 数据集名称
            force_redownload: 是否强制重新下载
        
        Returns:
            数据集路径，如果失败返回 None
        """
        if not BEIR_AVAILABLE:
            print(f"❌ beir 库未安装，无法下载 {dataset_name}")
            print("   请安装: pip install beir")
            return None
        
        dataset_dir = self.base_data_dir / "beir" / dataset_name
        
        # 检查是否已存在
        if dataset_dir.exists() and not force_redownload:
            corpus_file = dataset_dir / "corpus.jsonl"
            queries_file = dataset_dir / "queries.jsonl"
            qrels_file = dataset_dir / "qrels" / "test.tsv"
            
            if corpus_file.exists() and queries_file.exists() and qrels_file.exists():
                print(f"✓ {dataset_name} 已存在，跳过下载: {dataset_dir}")
                return str(dataset_dir)
        
        try:
            print(f"📥 下载 {dataset_name}...")
            url = self.BEIR_BASE_URL.format(dataset_name)
            out_dir = self.base_data_dir / "beir"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            data_path = util.download_and_unzip(url, str(out_dir))
            print(f"✓ {dataset_name} 下载完成: {data_path}")
            return data_path
        except Exception as e:
            print(f"❌ {dataset_name} 下载失败: {e}")
            return None
    
    def download_all_beir_datasets(self, force_redownload: bool = False) -> Dict[str, Optional[str]]:
        """
        下载所有 BEIR 数据集
        
        Args:
            force_redownload: 是否强制重新下载
        
        Returns:
            数据集名称到路径的映射
        """
        results = {}
        beir_datasets = self.supported_datasets['beir_datasets']
        
        print(f"\n开始下载 {len(beir_datasets)} 个 BEIR 数据集...")
        print("=" * 60)
        
        for dataset_name in tqdm(beir_datasets, desc="下载进度"):
            result = self.download_beir_dataset(dataset_name, force_redownload)
            results[dataset_name] = result
        
        print("\n" + "=" * 60)
        successful = sum(1 for v in results.values() if v is not None)
        print(f"下载完成: {successful}/{len(beir_datasets)} 成功")
        
        return results
    
    def download_trec_dl_dataset(self, year: str = "2019", force_redownload: bool = False) -> Optional[str]:
        """
        下载 TREC DL 数据集
        
        Args:
            year: 年份 ("2019" 或 "2020")
            force_redownload: 是否强制重新下载
        
        Returns:
            数据集路径，如果失败返回 None
        """
        dataset_name = f"trec-dl-{year}"
        dataset_dir = self.base_data_dir / dataset_name
        
        # 检查是否已存在
        if dataset_dir.exists() and not force_redownload:
            corpus_file = dataset_dir / "corpus.jsonl"
            queries_file = dataset_dir / "queries.jsonl"
            qrels_file = dataset_dir / "qrels" / "test.jsonl"
            
            if corpus_file.exists() and queries_file.exists() and qrels_file.exists():
                print(f"✓ {dataset_name} 已存在，跳过下载: {dataset_dir}")
                return str(dataset_dir)
        
        print(f"⚠️  {dataset_name} 需要手动下载")
        print(f"   请访问: https://microsoft.github.io/msmarco/TREC-Deep-Learning.html")
        print(f"   下载后解压到: {dataset_dir}")
        return None
    
    def download_all(self, 
                     include_beir: bool = True,
                     include_trec_dl: bool = True,
                     force_redownload: bool = False) -> Dict[str, Dict[str, Optional[str]]]:
        """
        下载所有数据集
        
        Args:
            include_beir: 是否包含 BEIR 数据集
            include_trec_dl: 是否包含 TREC DL 数据集
            force_redownload: 是否强制重新下载
        
        Returns:
            下载结果字典
        """
        results = {
            'beir': {},
            'trec_dl': {},
        }
        
        print("=" * 60)
        print("数据集下载器")
        print("=" * 60)
        print(f"基础数据目录: {self.base_data_dir.absolute()}")
        print()
        
        # 下载 BEIR 数据集
        if include_beir:
            results['beir'] = self.download_all_beir_datasets(force_redownload)
        
        # 下载 TREC DL 数据集
        if include_trec_dl:
            print("\n" + "=" * 60)
            print("TREC DL 数据集")
            print("=" * 60)
            for year in ['2019', '2020']:
                results['trec_dl'][f'trec-dl-{year}'] = self.download_trec_dl_dataset(year, force_redownload)
        
        # 打印汇总
        print("\n" + "=" * 60)
        print("下载汇总")
        print("=" * 60)
        
        total_success = 0
        total_count = 0
        
        if results['beir']:
            success = sum(1 for v in results['beir'].values() if v is not None)
            total = len(results['beir'])
            print(f"BEIR 数据集: {success}/{total} 成功")
            total_success += success
            total_count += total
        
        if results['trec_dl']:
            success = sum(1 for v in results['trec_dl'].values() if v is not None)
            total = len(results['trec_dl'])
            print(f"TREC DL 数据集: {success}/{total} 成功")
            total_success += success
            total_count += total
        
        print(f"\n总计: {total_success}/{total_count} 成功")
        
        return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download BEIR data and print TREC-DL placement instructions.")
    parser.add_argument(
        '--data_dir',
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help='数据存储目录（默认: RERANK_BIAS_DATA_ROOT 或 ./data）'
    )
    parser.add_argument(
        '--include_beir',
        action='store_true',
        default=True,
        help='包含 BEIR 数据集（默认: True）'
    )
    parser.add_argument(
        '--no_beir',
        action='store_true',
        help='不包含 BEIR 数据集'
    )
    parser.add_argument(
        '--include_trec_dl',
        action='store_true',
        default=True,
        help='包含 TREC DL 数据集（默认: True）'
    )
    parser.add_argument(
        '--no_trec_dl',
        action='store_true',
        help='不包含 TREC DL 数据集'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新下载已存在的数据集'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='指定要下载的数据集名称（仅用于 BEIR 数据集）'
    )
    
    args = parser.parse_args()
    
    # 处理互斥参数
    include_beir = args.include_beir and not args.no_beir
    include_trec_dl = args.include_trec_dl and not args.no_trec_dl
    
    downloader = DatasetDownloader(base_data_dir=args.data_dir)
    
    # 如果指定了特定数据集，只下载这些
    if args.datasets:
        print(f"下载指定的数据集: {args.datasets}")
        results = {}
        for dataset_name in args.datasets:
            if dataset_name in downloader.supported_datasets['beir_datasets']:
                result = downloader.download_beir_dataset(dataset_name, args.force)
                results[dataset_name] = result
            else:
                print(f"⚠️  未知的数据集: {dataset_name}")
        print(f"\n完成: {sum(1 for v in results.values() if v is not None)}/{len(results)} 成功")
    else:
        # 下载所有数据集
        downloader.download_all(
            include_beir=include_beir,
            include_trec_dl=include_trec_dl,
            force_redownload=args.force
        )


if __name__ == "__main__":
    main()
