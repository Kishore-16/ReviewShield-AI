"""
Dataset Merger & Global Shuffler Engine.
Combines 60,000 OR reviews and 60,000 CG reviews into a 120,000-row globally shuffled benchmark dataset.
"""

import csv
import random
from pathlib import Path
from typing import List, Dict, Tuple
from .models import ORRow, CGRow
from .serializer import DatasetSerializer, HEADER_COLUMNS


class DatasetMerger:
    """Merges OR and CG datasets across categories and applies global random shuffling."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.serializer = DatasetSerializer()

    def merge_and_shuffle(
        self,
        or_dir: Path,
        cg_dir: Path,
        output_file: Path
    ) -> Tuple[int, int, Dict[str, Tuple[int, int]]]:
        """Merges all OR and CG CSV files in directories, shuffles globally, and writes to output_file.
        
        Returns:
            (total_or_count, total_cg_count, category_breakdown)
        """
        all_rows: List[List[str]] = []
        or_count = 0
        cg_count = 0
        category_breakdown: Dict[str, Tuple[int, int]] = {}

        # 1. Gather all OR files
        or_files = sorted(list(Path(or_dir).glob("*_down_final_labeled.csv")))
        for f in or_files:
            rows = self.serializer.read_or_csv(f)
            for r in rows:
                all_rows.append([r.category, f"{r.rating:.1f}", r.label, r.text_])
                or_count += 1
                cat_or, cat_cg = category_breakdown.get(r.category, (0, 0))
                category_breakdown[r.category] = (cat_or + 1, cat_cg)

        # 2. Gather all CG files
        cg_files = sorted(list(Path(cg_dir).glob("*.csv")))
        for f in cg_files:
            if f == output_file or f.name.endswith("_merged.csv"):
                continue
            with open(f, mode="r", encoding="utf-8", newline="") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    cat = row.get("category", "")
                    rat = float(row.get("rating", 5.0))
                    lbl = row.get("label", "CG")
                    txt = row.get("text_", "")
                    all_rows.append([cat, f"{rat:.1f}", lbl, txt])
                    cg_count += 1
                    cat_or, cat_cg = category_breakdown.get(cat, (0, 0))
                    category_breakdown[cat] = (cat_or, cat_cg + 1)

        # 3. Apply global random shuffle
        rng = random.Random(self.random_seed)
        rng.shuffle(all_rows)

        # 4. Write merged shuffled dataset
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, mode="w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(HEADER_COLUMNS)
            writer.writerows(all_rows)

        return or_count, cg_count, category_breakdown
