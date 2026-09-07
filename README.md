# ranking-recovery-creativity

Code, processed results, and redistributable study files for ranking-recovery
creativity evaluation.

Original Reedsy story texts are not redistributed. Source pages are identified
by URL in `data/source_manifest.csv`. Ranking uses extracted idea representations
(`extracted_idea_250`) and vote-derived reference orderings.

## Layout

```text
ranking-recovery-creativity/
├── README.md
├── LICENSE
├── code/                 # experiment scripts
├── src/                  # installable Python package
├── configs/              # YAML configs used in the paper
├── tests/
├── data/
│   ├── source_manifest.csv
│   ├── extracted_ideas.csv
│   ├── votes.csv
│   └── reference_rankings.csv
├── saida_final.json      # nested dataset without original story texts
├── results/
└── ...
```

## Public data files

- `data/source_manifest.csv` - the 20 Reedsy prompt URLs used in the study
- `data/extracted_ideas.csv` - idea representations (50 / 150 / 250 words)
- `data/votes.csv` - popularity votes (`likes`) and `comments`
- `data/reference_rankings.csv` - vote-derived rank within each prompt (1 = most likes)
- `saida_final.json` - same records in the nested format expected by the loader,
  with original `Content` fields removed

Join key: `story_url` within `prompt_id`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional API keys (only to re-run embedding/API models, not to inspect shipped CSVs):

- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`

Load them from a local `.env` (never commit that file).

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

Example:

```bash
python code/run_h2_batch.py --config configs/h2_batch_paper_models.yaml
```

## Notes

- Shipped `results/` are enough to verify the paper tables.
- Re-running LLM judges from raw GVALD parquet dumps requires `LLM-as-judge-data-geval/`
  (not included).
- Re-running some baseline scripts may require intermediate `*_detalhado.json`
  folders that are also omitted from this package.
