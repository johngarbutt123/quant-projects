from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class AlignmentAudit:
    n_rows_a: int
    n_rows_b: int
    n_rows_common: int
    n_cols_a: int
    n_cols_b: int
    n_cols_common: int
    dates_only_in_a: pd.Index
    dates_only_in_b: pd.Index
    cols_only_in_a: pd.Index
    cols_only_in_b: pd.Index

    def summary(self, name_a: str = "A", name_b: str = "B", max_cols: int = 20) -> str:
        lines = []
        lines.append("=== Alignment audit ===")
        lines.append(
            f"{name_a} dates: {self.n_rows_a:,} | {name_b} dates: {self.n_rows_b:,} | Common: {self.n_rows_common:,}"
        )
        lines.append(
            f"{name_a} cols : {self.n_cols_a:,} | {name_b} cols : {self.n_cols_b:,} | Common: {self.n_cols_common:,}"
        )

        lines.append(f"\nDates only in {name_a}: {len(self.dates_only_in_a):,}")
        if len(self.dates_only_in_a) > 0:
            lines.append(
                f"  first: {self.dates_only_in_a.min()}  last: {self.dates_only_in_a.max()}"
            )

        lines.append(f"Dates only in {name_b}: {len(self.dates_only_in_b):,}")
        if len(self.dates_only_in_b) > 0:
            lines.append(
                f"  first: {self.dates_only_in_b.min()}  last: {self.dates_only_in_b.max()}"
            )

        lines.append(f"\nTickers only in {name_a}: {len(self.cols_only_in_a):,}")
        if len(self.cols_only_in_a) > 0:
            lines.append(
                f"  {list(self.cols_only_in_a[:max_cols])}"
                + (" ..." if len(self.cols_only_in_a) > max_cols else "")
            )

        lines.append(f"Tickers only in {name_b}: {len(self.cols_only_in_b):,}")
        if len(self.cols_only_in_b) > 0:
            lines.append(
                f"  {list(self.cols_only_in_b[:max_cols])}"
                + (" ..." if len(self.cols_only_in_b) > max_cols else "")
            )

        return "\n".join(lines)


def align_on_index_and_columns(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    name_a: str = "A",
    name_b: str = "B",
    sort_index: bool = True,
    audit: bool = False,
    require_nonempty: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, AlignmentAudit]:
    """
    Align two DataFrames to common index and columns.
    Returns aligned copies + an audit object (always returned).
    """
    idx_common = a.index.intersection(b.index)
    cols_common = a.columns.intersection(b.columns)

    audit_obj = AlignmentAudit(
        n_rows_a=len(a.index),
        n_rows_b=len(b.index),
        n_rows_common=len(idx_common),
        n_cols_a=len(a.columns),
        n_cols_b=len(b.columns),
        n_cols_common=len(cols_common),
        dates_only_in_a=a.index.difference(b.index),
        dates_only_in_b=b.index.difference(a.index),
        cols_only_in_a=a.columns.difference(b.columns),
        cols_only_in_b=b.columns.difference(a.columns),
    )

    if require_nonempty and (len(idx_common) == 0 or len(cols_common) == 0):
        raise ValueError(
            f"No overlap when aligning {name_a} vs {name_b}: "
            f"common rows={len(idx_common)}, common cols={len(cols_common)}"
        )

    a_aligned = a.loc[idx_common, cols_common]
    b_aligned = b.loc[idx_common, cols_common]

    if sort_index:
        a_aligned = a_aligned.sort_index()
        b_aligned = b_aligned.sort_index()

    # Hard guarantees (avoid silent issues later)
    if not a_aligned.index.equals(b_aligned.index):
        raise AssertionError("Aligned indexes do not match.")
    if not (a_aligned.columns == b_aligned.columns).all():
        raise AssertionError("Aligned columns do not match.")

    if audit:
        print(audit_obj.summary(name_a=name_a, name_b=name_b))

    return a_aligned, b_aligned, audit_obj
