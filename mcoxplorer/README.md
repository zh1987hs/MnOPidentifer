# MCOxplorer

MCOxplorer is an offline, reproducible **sequence + structure multimodal candidate prioritization** toolkit for remote Mn(II)-oxidizing multicopper oxidase (MCO) discovery.

> Candidate prioritization tool only; not final functional annotation.

## Real implementation vs fallback
### Real implementation (default priority)
- MMseqs2 sequence search.
- Positive-sequence clustering: MMseqs2 easy-cluster (fallback: greedy identity clustering).
- HMMER profile scoring with **MSA-first** pipeline: MAFFT (preferred) or MUSCLE -> hmmbuild -> hmmsearch.
- ESM2 local embedding runtime with one-time model/tokenizer init and batch inference.
- Optional structure-aware embedding runtime (ProstT5 preferred; ESM2/mock fallback supported by config).
- Foldseek structure search + optional TM-align refinement.
- Structure QC with PDB/mmCIF parsing + metadata fusion + quality thresholds.
- Local support includes sequence motif heuristics + optional 3D neighborhood stats.

### Fallback behavior
- Missing MMseqs2/HMMER/MSA/ESM2/TM-align gracefully degrades with explicit backend labels.
- Missing Foldseek downgrades structure workflow unless `runtime.structure_required=true`.

## Dependencies
Python package deps are in `pyproject.toml`.
External tools:
- MMseqs2
- HMMER (`hmmbuild`, `hmmsearch`)
- MAFFT (recommended) or MUSCLE
- Foldseek
- TM-align

Configure executable paths in `config/default.yaml` under `external_tools`.

## Run
```bash
mcoxplorer run -c config/default.yaml
```

## Key outputs
- `candidate_sequence_features.csv`
- `candidate_structure_features.csv`
- `ranked_candidates_sequence_view.csv`
- `ranked_candidates_structure_view.csv`
- `ranked_candidates_multimodal_view.csv`
- `id_mapping_resolved.csv`
- `structure_qc_summary.csv`
- `reports/top_candidates_report.md`
- `reports/positive_structure_cluster_report.md`

## Local feature interpretation
- `sequence_local_support`: sequence-neighborhood heuristic support.
- `structure_local_support_3d`: coordinate-based local neighborhood support (if motif mapping succeeds).
- `combined_local_support`: fused local support used in scoring.

## Practical notes
- `sequence_cluster_backend` indicates `mmseqs2` or `fallback_greedy`.
- `structure_aware_embedding_backend` indicates which structure-aware embedding backend actually ran (`prostt5`, `esm2`, `mock`, or `disabled`).
- `structure_quality_penalty` and `structure_evidence_usable` directly affect structure score and report interpretation.
- `remote_but_structure_supported` and `high_confidence_first_batch` help experiment prioritization.

## Structure-aware embedding (optional)
Under `sequence.structure_aware_embedding` in `config/default.yaml`:
- `enabled`: turn this feature stream on/off.
- `backend`: `prostt5`, `esm2`, or `mock`.
- `model_name`, `cache_dir`, `device`, `batch_size`, `force_mock`: runtime controls.

Feature outputs in `candidate_sequence_features.csv`:
- `structure_aware_embedding_backend`
- `structure_aware_distance_to_positive_centroid`
- `structure_aware_novelty`
- `nearest_positive_structure_aware_distance`

Scoring integration is opt-in via `sequence.structure_aware_embedding_weight` (default `0.0`, i.e., no impact on sequence score).

## Limitations
- Local 3D features are first-pass spatial statistics (not full pocket energetics/MD).
- Structure similarity does not guarantee Mn(II)-oxidizing activity.
- Wet-lab validation is required.


## Structure quality penalty config
In `config/default.yaml -> structure.quality`:
- `penalty_if_quality_fail`
- `penalty_if_low_coverage`
- `penalty_if_low_confidence`

These values directly down-weight `structure_score`, and therefore affect fused ranking.

## Motif-to-structure mapping boundary
3D local support uses a first-pass residue-order mapping with tolerance, then computes neighborhood statistics around mapped motif residues.
If mapping is insufficient, local support falls back to sequence heuristic and report marks this source explicitly.
