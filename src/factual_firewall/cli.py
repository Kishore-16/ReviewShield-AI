"""
Command Line Interface (CLI) for Factual Firewall Dataset Engine.
"""

import argparse
import sys
from pathlib import Path

from .config import EngineConfig
from .pipeline import FactualFirewallPipeline
from .merger import DatasetMerger
from .serializer import DatasetSerializer
from .validator import QualityAssuranceValidator


def main():
    parser = argparse.ArgumentParser(
        description="Factual Firewall Dataset Engine - Synthetic CG Review Generator"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command 1: generate
    gen_parser = subparsers.add_parser("generate", help="Generate CG reviews for a single OR file")
    gen_parser.add_argument("--input", "-i", type=str, required=True, help="Input OR CSV file path")
    gen_parser.add_argument("--output", "-o", type=str, required=True, help="Output CG CSV file path")
    gen_parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of rows for test/dry-run")
    gen_parser.add_argument("--label", default="CG", help="Target CG label format ('CG' or '1')")

    # Command 2: generate-all
    gen_all_parser = subparsers.add_parser("generate-all", help="Generate CG reviews for all 12 input OR datasets")
    gen_all_parser.add_argument("--input-dir", default=".", help="Directory containing input OR files")
    gen_all_parser.add_argument("--output-dir", default="./cg_outputs", help="Directory to save CG files")
    gen_all_parser.add_argument("--limit", type=int, default=None, help="Limit per file (for testing)")
    gen_all_parser.add_argument("--label", default="CG", help="Target CG label format ('CG' or '1')")

    # Command 3: merge
    merge_parser = subparsers.add_parser("merge", help="Merge OR and CG datasets with global shuffling")
    merge_parser.add_argument("--or-dir", default=".", help="Directory containing original OR datasets")
    merge_parser.add_argument("--cg-dir", default="./cg_outputs", help="Directory containing CG datasets")
    merge_parser.add_argument("--output", default="./final_120k_or_cg_dataset.csv", help="Path for final merged dataset")
    merge_parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")

    # Command 4: validate
    val_parser = subparsers.add_parser("validate", help="Validate a generated CG or merged dataset file")
    val_parser.add_argument("--file", "-f", type=str, required=True, help="Path to CSV file to validate")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        config = EngineConfig()
        config.pipeline.label_format = args.label
        
        pipeline = FactualFirewallPipeline(config)
        print(f"[*] Starting CG generation for '{input_path.name}'...")
        cg_rows = pipeline.process_file_sync(input_path, output_path, limit=args.limit)
        print(f"[OK] Successfully generated {len(cg_rows)} CG rows -> '{output_path}'")

    elif args.command == "generate-all":
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config = EngineConfig()
        config.pipeline.label_format = args.label
        pipeline = FactualFirewallPipeline(config)

        or_files = sorted(list(input_dir.glob("*_down_final_labeled.csv")))
        print(f"[*] Found {len(or_files)} OR dataset files in '{input_dir}'")

        total_cg = 0
        for f in or_files:
            output_file = output_dir / f.name.replace("_down_final_labeled.csv", "_cg_labeled.csv")
            print(f"\n[*] Processing '{f.name}' -> '{output_file.name}'...")
            cg_rows = pipeline.process_file_sync(f, output_file, limit=args.limit)
            total_cg += len(cg_rows)
            print(f"[OK] Category '{f.name}' complete: {len(cg_rows)} rows.")

        print(f"\n[OK] ALL BATCH GENERATION COMPLETE! Total CG reviews generated: {total_cg}")

    elif args.command == "merge":
        or_dir = Path(args.or_dir)
        cg_dir = Path(args.cg_dir)
        output_file = Path(args.output)

        print(f"[*] Merging OR datasets from '{or_dir}' and CG datasets from '{cg_dir}'...")
        merger = DatasetMerger(random_seed=args.seed)
        or_count, cg_count, breakdown = merger.merge_and_shuffle(or_dir, cg_dir, output_file)

        print(f"\n[OK] MERGE & GLOBAL SHUFFLE COMPLETE!")
        print(f"    - Total OR Rows: {or_count}")
        print(f"    - Total CG Rows: {cg_count}")
        print(f"    - Total Dataset Rows: {or_count + cg_count}")
        print(f"    - Output File: {output_file.resolve()}")
        print("\n    Category Breakdown:")
        for cat, (o_c, c_c) in breakdown.items():
            print(f"      - {cat}: {o_c} OR + {c_c} CG = {o_c + c_c} total")

    elif args.command == "validate":
        target_file = Path(args.file)
        if not target_file.exists():
            print(f"[X] File not found: {target_file}")
            sys.exit(1)

        print(f"[*] Validating dataset: '{target_file}'...")
        validator = QualityAssuranceValidator()
        serializer = DatasetSerializer()

        or_rows = serializer.read_or_csv(target_file)
        print(f"    - Total rows loaded: {len(or_rows)}")
        
        valid_count = 0
        preambles = 0
        markdowns = 0

        for r in or_rows:
            cleaned, preamble, markdown = validator.sanitize_text(r.text_)
            if preamble:
                preambles += 1
            if markdown:
                markdowns += 1
            if len(cleaned.split()) >= 3 and not preamble and not markdown:
                valid_count += 1

        print(f"\n[OK] Validation Summary:")
        print(f"    - Total Rows: {len(or_rows)}")
        print(f"    - Clean Valid Rows: {valid_count} ({(valid_count/len(or_rows))*100:.1f}%)")
        print(f"    - Preambles Detected: {preambles}")
        print(f"    - Markdown Formatting Detected: {markdowns}")


if __name__ == "__main__":
    main()
