# 6-week internship roadmap

1. **Week 1 — Dataset & problem framing:** download one CIC-IDS2017 CSV; document dataset version, label distribution, feature schema, license/terms, and threat model.
2. **Week 2 — Data engineering:** clean whitespace in labels/columns, convert infinities to null, remove duplicates carefully, create reproducible train/validation/test splits, save clean Parquet.
3. **Week 3 — Baselines:** compare DummyClassifier, Random Forest, Extra Trees, and XGBoost; report macro-F1, per-class precision/recall, confusion matrix, and PR-AUC for binary detection.
4. **Week 4 — Explainability:** use SHAP on a bounded sample; inspect false negatives and feature stability; note that feature importance is not causality.
5. **Week 5 — Serving:** wrap the saved pipeline in FastAPI; add a Streamlit dashboard showing traffic summaries, class counts, confidence, and explanations. Validate feature schema.
6. **Week 6 — Scale & deliver:** process larger data with Spark and Parquet; optionally add Kafka only if a real streaming source is available; add Docker, tests, screenshots, setup steps, limitations, and a short demo video.

## 16 GB RAM strategy
- Start with 50k–200k rows and a single CSV.
- Use numeric-only feature matrices and float32 where suitable.
- Avoid `n_jobs=-1` for every tool at once; keep XGBoost threads modest.
- Do not read all 60–100 GB of raw files into pandas. Use Spark/Dask or partitioned Parquet for larger workloads.
- Never report synthetic-data scores as real IDS performance.

## Completion checklist
- [ ] Dataset source and exact file named in README
- [ ] Reproducible split strategy documented
- [ ] Baseline comparison table and per-class report saved
- [ ] False-negative analysis completed
- [ ] SHAP/permutation plots saved
- [ ] Model artifact and feature schema versioned
- [ ] API/dashboard tested on held-out data
- [ ] Resume claims match measured results
