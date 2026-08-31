# ranking-recovery-creativity

Code, dataset, and machine-readable results for the ranking-recovery creativity evaluation experiments.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional API keys (only needed to re-run embedding/API models, not to inspect shipped CSVs):

- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`

Load them from a local `.env` (never commit that file).

## Layout

- `src/embedding_models_eval/` - pipeline, metrics, H1/H2, LLM-judge normalization
- `scripts/` - experiment entry points
- `configs/` - YAML configs used for the paper runs
- `tests/` - unit/smoke tests
- `saida_final.json` - contest prompts, submissions, votes, extracted ideas
- `all_calls_extractor.json` - idea-extraction call logs
- `results/` - observed metrics and chance baselines used in the paper tables

## Inspect shipped results

Main comparison tables:

- `results/experimento2_h1_h2/comparacao_h1_h2.csv`
- `results/experimento2_h1_h2_topn_real_ratio/comparacao_h1_h2.csv`
- `results/llm_judge_normalized/latex_ready/`

Chance baselines (R=2500):

- `results/experimento2_h2_random_baseline_2500/summary.csv`
- `results/experimento2_h2_topn_real_ratio_random_baseline_2500/summary.csv`
- `results/experimento2_h1_topn_real_ratio_random_baseline_2500/summary.csv`
- `results/llm_judge_normalized/by_protocol/tournament_description/random_baseline_*_2500/summary.csv`

## Notes

- Shipped `results/` are enough to verify the paper tables.
- Re-running LLM judges from raw GVALD parquet dumps requires `LLM-as-judge-data-geval/` (not included in this minimal package).
- Re-running some baseline scripts may require intermediate `*_detalhado.json` folders that are also omitted from this minimal package.
