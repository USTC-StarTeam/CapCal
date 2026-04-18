# Data Loaders

The cleaned repository keeps only the data loaders required for the paper's main experiments:

- BEIR loaders, used for `trec-covid`
- TREC DL 2019/2020 loader
- TREC DL 2021/2022/2023 loader

## Supported Dataset Layout

```text
$RERANK_BIAS_DATA_ROOT/
├── beir/
│   └── trec-covid/
├── trec-dl-2019/
├── trec-dl-2020/
├── trec/
│   ├── trec21/
│   ├── trec22/
│   └── trec23/
└── msmarco_v2_passage/
```

`msmarco_v2_passage` is required by the TREC DL 2021/2022/2023 loader unless you override it with `RERANK_MSMARCO_V2_PASSAGE_DIR`.

## Factory Surface

`DataLoaderFactory` now supports only:

- `beir`
- `trec_dl`
- `trec_dl_212223`

This matches the reduced experiment scope of the repository.
