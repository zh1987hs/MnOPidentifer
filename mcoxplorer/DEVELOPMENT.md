# Development Notes

## v0.2 (Phase-2) architecture decisions
1. Add explicit Foldseek/TM-align runner+parser layers with reusable intermediate artifacts and graceful fallback.
2. Upgrade positive structure analysis to matrix-driven hierarchical clustering with medoid prototypes.
3. Keep sequence-only survivability: candidates without structures are retained and downgraded gracefully in fusion.
4. Keep fusion transparent and configurable (weighted additive scoring).
5. Use deterministic template rendering for experiment-facing reports.

## Current extension points
- Replace heuristic similarity transforms with direct Foldseek/TM-align calibrated normalization.
- Integrate full MMseqs2/HMMER/ESM2 production backends.
- Expand local metal-site environment descriptors.
- Add richer provenance and pipeline checkpointing.
