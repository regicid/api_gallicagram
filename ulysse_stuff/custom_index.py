

from typing import Iterable, Optional

from toolbox import DFs, index_author, init_tables
from pyalex_client import PyAlexClient
import pandas as pd




def _h_index(citations: Iterable[int]) -> int:
    """Standard h-index from a list/iterable of citation counts."""
    # defensive: convert to ints, drop NAs, ensure non-negative
    c = sorted((int(x) for x in citations if pd.notna(x)), reverse=True)
    h = 0
    for i, v in enumerate(c, 1):
        if v >= i:
            h = i
        else:
            break
    return h

def _h_index_for(author_id: str,
                 dfs: DFs,
                 *,
                 position: Optional[str] = None,   # "first" | "middle" | "last" | None
                 year_from: Optional[int] = None) -> int:
    """Compute h-index for one author, optionally filtered by role and year."""
    # slice authorship to this author (+ role if requested)
    mask = dfs.authorship["author_id"].eq(author_id)
    if position is not None:
        if position not in {"first","middle","last"}:
            raise ValueError("position must be one of {'first','middle','last', None}")
        mask &= dfs.authorship["position"].eq(position)

    paper_ids = dfs.authorship.loc[mask, "paper_id"].unique()
    if len(paper_ids) == 0:
        return 0

    # pull (year, cited_by_count) for those papers
    meta = dfs.papers.loc[paper_ids, ["year","cited_by_count"]].copy()

    # optional year filter (treat missing years as excluded)
    if year_from is not None:
        meta = meta[meta["year"].ge(year_from, fill_value=False)]
        if meta.empty:
            return 0

    # compute h on integer citation counts
    return _h_index(meta["cited_by_count"].fillna(0).astype(int).tolist())

def h_index_overall(author_id: str, dfs: DFs, *, year_from: Optional[int] = None) -> int:
    return _h_index_for(author_id, dfs, position=None, year_from=year_from)

def h_index_first(author_id: str, dfs: DFs, *, year_from: Optional[int] = None) -> int:
    return _h_index_for(author_id, dfs, position="first", year_from=year_from)

def h_index_middle(author_id: str, dfs: DFs, *, year_from: Optional[int] = None) -> int:
    return _h_index_for(author_id, dfs, position="middle", year_from=year_from)

def h_index_last(author_id: str, dfs: DFs, *, year_from: Optional[int] = None) -> int:
    return _h_index_for(author_id, dfs, position="last", year_from=year_from)


def print_h_index(author_id: str, dfs: DFs, *, year_from: Optional[int] = None) -> None:
    """Pretty-print h-index values for one author."""
    if author_id not in dfs.authors.index:
        raise ValueError(f"Author {author_id} not found in authors table.")
    
    fname = dfs.authors.loc[author_id, "first_name"] or ""
    lname = dfs.authors.loc[author_id, "last_name"] or ""
    full_name = f"{fname} {lname}".strip()

    h_all = h_index_overall(author_id, dfs, year_from=year_from)
    h_f   = h_index_first(author_id, dfs, year_from=year_from)
    h_m   = h_index_middle(author_id, dfs, year_from=year_from)
    h_l   = h_index_last(author_id, dfs, year_from=year_from)

    yr_info = f" since {year_from}" if year_from is not None else ""
    print(f"{full_name} ({author_id}) — h-index{yr_info}: all={h_all}, first={h_f}, middle={h_m}, last={h_l}")


def get_all_h_indexes(author_id: str, dfs = None, year_from = None) -> dict:
    if dfs is None:
        dfs = DFs.from_tuple(init_tables())

    pyalex_client = PyAlexClient()

    index_author(author_id, dfs, pyalex_client)

    return {
        "overall": h_index_overall(author_id, dfs, year_from=year_from),
        "first":   h_index_first(author_id, dfs, year_from=year_from),
        "middle":  h_index_middle(author_id, dfs, year_from=year_from),
        "last":    h_index_last(author_id, dfs, year_from=year_from),
    }
