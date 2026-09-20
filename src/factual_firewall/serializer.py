"""
CSV Reader and Serializer Engine for Stage 6 Dataset Serialization.
"""

import csv
from pathlib import Path
from typing import List, Generator
from .models import ORRow, CGRow


HEADER_COLUMNS = ["category", "rating", "label", "text_"]


class DatasetSerializer:
    """Handles high-performance streaming CSV reads and writes following strict schema."""

    @staticmethod
    def read_or_csv(file_path: Path) -> List[ORRow]:
        """Reads Original Review (OR) CSV file into a list of ORRow models."""
        rows: List[ORRow] = []
        with open(file_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle possible alternative column names
                cat = row.get("category") or row.get("Category", "")
                rat = float(row.get("rating") or row.get("Rating", 5.0))
                lbl = row.get("label") or row.get("Label", "OR")
                txt = row.get("text_") or row.get("text") or row.get("Text", "")
                
                rows.append(ORRow(category=cat, rating=rat, label=lbl, text_=txt))
        return rows

    @staticmethod
    def write_cg_csv(file_path: Path, cg_rows: List[CGRow], append: bool = False):
        """Serializes CG rows to CSV with proper quoting and 4-column schema."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        
        write_header = not append or not file_path.exists() or file_path.stat().st_size == 0
        
        with open(file_path, mode=mode, encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            if write_header:
                writer.writerow(HEADER_COLUMNS)
            
            for r in cg_rows:
                writer.writerow([r.category, f"{r.rating:.1f}", r.label, r.text_])

    @staticmethod
    def stream_or_csv(file_path: Path) -> Generator[ORRow, None, None]:
        """Streams OR CSV row by row to conserve memory on large files."""
        with open(file_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat = row.get("category") or row.get("Category", "")
                rat = float(row.get("rating") or row.get("Rating", 5.0))
                lbl = row.get("label") or row.get("Label", "OR")
                txt = row.get("text_") or row.get("text") or row.get("Text", "")
                yield ORRow(category=cat, rating=rat, label=lbl, text_=txt)
