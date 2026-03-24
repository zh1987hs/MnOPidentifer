# Development Notes

## v0.3 (real-first sequence + strengthened structure)
1. Sequence module now prioritizes real integrations: MMseqs2, HMMER, and local ESM2 cache.
2. All sequence real integrations keep deterministic fallback paths with explicit backend labels.
3. Structure QC now fuses parsed metrics and metadata-based coverage/confidence with configurable quality gates.
4. Added local motif-neighborhood structural heuristics for first-pass Mn-related support.
5. Fusion adds explicit `remote_but_structure_supported` and `high_confidence_first_batch` recommendation fields.

## Remaining advanced work
- Calibrate MMseqs/Foldseek/TM-align/HMM scores on curated benchmark sets.
- Integrate true MSA-based HMM construction and profile management.
- Expand local-site descriptors to geometric/environmental features.
