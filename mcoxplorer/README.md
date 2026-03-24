# MCOxplorer

MCOxplorer is an offline, reproducible **sequence + structure multimodal candidate prioritization** toolkit for remote Mn(II)-oxidizing multicopper oxidase (MCO) discovery.

> **Important:** This project is a candidate-ranking and experiment-guidance tool, **not** a final function annotator.

## Scientific positioning
- Focus: identify which unannotated proteins are most worth wet-lab validation.
- Key value: rescue **sequence-remote but structure-supported** candidates.
- Output style: transparent scores + rule-based Markdown reports for experimental teams.

## Core modules
- `sequence`: FASTA QC, positive clustering, sequence similarity, embedding novelty.
- `structure`: Foldseek/TM-align interfaces, positive structure clustering, prototype selection, candidate structural features.
- `fusion`: configurable multimodal scoring/ranking.
- `reports`: deterministic template-based report generation (no online LLM API).

## Install
```bash
conda env create -f environment.yml
conda activate mcoxplorer
pip install -e .[dev]
```

## External dependencies
Optional but recommended:
- **Foldseek** (required for full structure workflow)
- **TM-align** (optional refinement for top hits)
- MMseqs2 (future sequence acceleration path)

You can provide executable names in `config/default.yaml` under `external_tools`.

## Quick start
```bash
mcoxplorer init-example --dest .
mcoxplorer validate-config -c config/default.yaml
mcoxplorer run -c config/default.yaml
```

## Structure-only run
```bash
mcoxplorer structure-only -c config/default.yaml
```

## CLI
- `init-example`
- `run`
- `sequence-only`
- `structure-only`
- `report`
- `validate-config`

## Key outputs
- `ranked_candidates_sequence_view.csv`
- `ranked_candidates_structure_view.csv`
- `ranked_candidates_multimodal_view.csv`
- `candidate_sequence_features.csv`
- `candidate_structure_features.csv`
- `positive_structure_similarity_matrix.csv`
- `positive_structure_clusters.csv`
- `positive_structure_prototypes.csv`
- `positive_structure_cluster_family_summary.csv`
- `structure_qc_summary.csv`
- `run_summary.json`
- `reports/top_candidates_report.md`
- `reports/positive_structure_cluster_report.md`

## Result interpretation hints
- High sequence + high structure: strong first-batch candidates.
- Low sequence + high structure: high-value remote candidates.
- High generic MCO risk: likely false-positive context; validate carefully.
- Missing structure: candidate kept, but fusion degrades toward sequence evidence.

## Limitations
- Structural similarity does **not** prove Mn(II)-oxidizing activity.
- Predicted models can be wrong in local regions.
- Current local-site chemistry features are heuristic-level only.
- Wet-lab validation remains required.
