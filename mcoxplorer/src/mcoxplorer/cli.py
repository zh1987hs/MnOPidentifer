from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mcoxplorer.reports.generator import generate_markdown_report, generate_positive_cluster_report
from mcoxplorer.sequence.pipeline import run_sequence_module
from mcoxplorer.structure.pipeline import run_structure_module
from mcoxplorer.utils.config import load_config, validate_config
from mcoxplorer.utils.logging import setup_logging
from mcoxplorer.workflows.run_pipeline import run_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mcoxplorer")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-example", help="Copy toy example input and config")
    p_init.add_argument("--dest", default=".")

    for name in ["run", "sequence-only", "structure-only", "report", "validate-config"]:
        sp = sub.add_parser(name)
        sp.add_argument("-c", "--config", default="config/default.yaml")

    return p


def cmd_init_example(dest: str) -> None:
    src = Path(__file__).resolve().parents[2] / "examples"
    cfg = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    dst = Path(dest)
    (dst / "examples").mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst / "examples", dirs_exist_ok=True)
    (dst / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg, dst / "config" / "default.yaml")
    print(f"Example initialized under: {dst}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "init-example":
        cmd_init_example(args.dest)
        return

    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))

    if args.cmd == "validate-config":
        validate_config(cfg)
        print("Config is valid")
    elif args.cmd == "run":
        run_all(cfg)
    elif args.cmd == "sequence-only":
        run_sequence_module(cfg)
    elif args.cmd == "structure-only":
        run_structure_module(cfg)
    elif args.cmd == "report":
        import pandas as pd

        ranked = pd.read_csv(Path(cfg["output_dir"]) / "ranked_candidates_multimodal_view.csv")
        generate_markdown_report(cfg, ranked)

        cluster_file = Path(cfg["output_dir"]) / "positive_structure_clusters.csv"
        proto_file = Path(cfg["output_dir"]) / "positive_structure_prototypes.csv"
        if cluster_file.exists() and proto_file.exists():
            clusters = pd.read_csv(cluster_file)
            protos = pd.read_csv(proto_file)
            generate_positive_cluster_report(cfg, clusters, protos)


if __name__ == "__main__":
    main()
