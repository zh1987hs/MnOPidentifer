# MCOxplorer

MCOxplorer is an offline, reproducible **sequence + structure multimodal candidate prioritization** toolkit for remote Mn(II)-oxidizing multicopper oxidase (MCO) discovery.

> **Important:** This project is a candidate-ranking and experiment-guidance tool, **not** a final function annotator.

## Current implementation status
### Real-first paths (default)
- Sequence similarity: **MMseqs2** search backend (fallback to in-memory pairwise identity).
- Profile models: **HMMER (hmmbuild+hmmsearch)** cluster profiles (fallback to constant neutral score).
- Embeddings: **ESM2 local cache** (fallback to deterministic mock embedding).
- Structure search: **Foldseek** (required for full structure mode).
- Structural refinement: **TM-align** optional top-hit refinement.

### Fallback/optional paths
- Missing MMseqs2/HMMER/ESM2/TM-align uses deterministic fallback with clear backend labels.
- Missing Foldseek triggers graceful structure downgrade unless `runtime.structure_required=true`.

## Install
```bash
conda env create -f environment.yml
conda activate mcoxplorer
pip install -e .[dev]
```

## External tool installation (example)
- MMseqs2: https://github.com/soedinglab/MMseqs2
- HMMER: http://hmmer.org/
- Foldseek: https://github.com/steineggerlab/foldseek
- TM-align: https://zhanggroup.org/TM-align/

Set executable names/paths in `config/default.yaml -> external_tools`.

## Run modes
### Full real mode (recommended)
```bash
mcoxplorer run -c config/default.yaml
```

### Lightweight fallback mode
Disable/omit external tools and keep `runtime.structure_required=false`.

### Structure-only
```bash
mcoxplorer structure-only -c config/default.yaml
```

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

## Interpretation highlights
- `remote_but_structure_supported=true`: sequence-remote but structure-supported high-value candidates.
- `high_confidence_first_batch=true`: recommended first-batch wet-lab candidates.
- `false_positive_risk` helps identify generic-MCO-like risk.
- Local motif/acidity support is provided in `candidate_structure_features.csv`.

## Limitations
- Structural similarity does **not** prove Mn(II)-oxidizing activity.
- Local motif features are first-version heuristics (not pocket energetics).
- Predicted models can be wrong in key regions.
- Wet-lab validation remains required.
