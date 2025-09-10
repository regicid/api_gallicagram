

from typing import Iterable, Optional

import pandas as pd
from dataclasses import dataclass
import pandas as pd

@dataclass
class DFs:
    authors:   pd.DataFrame
    papers:    pd.DataFrame
    citations: pd.DataFrame
    authorship:pd.DataFrame

    @classmethod
    def from_tuple(cls, dfs): return cls(*dfs)
    def as_tuple(self): return (self.authors, self.papers, self.citations, self.authorship)


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



from typing import Iterable, Optional

from time import sleep



class PyAlexClient:
    def fetch_author(self, author_id: str) -> dict:
        # TODO: retry/backoff if desired
        return Authors()[author_id]

    def iter_works_by_author(self, author_id: str, per_page=200):
        # Generator of work dicts
        for page in Works().filter(author={"id": author_id}).paginate(per_page=per_page):
            for w in page: yield w
            # sleep(0.1)  # Optional: be nice to the API


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








AUTHOR_COLS   = ["first_name","last_name","orcid"]
PAPER_COLS    = ["title","journal","year","cited_by_count","doi"]
CITE_COLS     = ["citing_paper","cited_paper"]
AUTHORSHIP_COLS = ["author_id","paper_id","position"]

def init_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    authors_df   = pd.DataFrame(index=pd.Index([], name="author_id"),
                                columns=AUTHOR_COLS, dtype="string")
    papers_df    = pd.DataFrame(index=pd.Index([], name="paper_id"),
                                columns=PAPER_COLS)
    citations_df = pd.DataFrame(columns=CITE_COLS)
    authorship_df= pd.DataFrame(columns=AUTHORSHIP_COLS)
    return authors_df, papers_df, citations_df, authorship_df


def _split_name(name: str) -> tuple[str, str]:
    """Split a name into first and last name, handling cases with no space."""
    parts = name.split(" ", 1)
    if len(parts) == 1:
        return parts[0], ""  # No space found, treat as first name only
    return parts[0], parts[1]  # First and last name

# ---- Empty frames with correct schema ----
def empty_coauthored_like():
    return pd.DataFrame(columns=["title","journal","year","cited_by_count","doi",
                                 "position_author_1","position_author_2"]) \
             .astype({"title":"string","journal":"string","doi":"string",
                      "year":"Int64","cited_by_count":"Int64",
                      "position_author_1":"string","position_author_2":"string"})

def empty_coauthor_detail_like():
    return pd.DataFrame(columns=["author_1_name","author_2_name","title","journal","year",
                                 "cited_by_count","doi","position_author_1","position_author_2"])

# ---- Validation ----
def ensure_loaded_author(author_id: str, df_authorship: pd.DataFrame):
    if author_id not in df_authorship["author_id"].values:
        raise ValueError("Author not loaded – run add_author() first.")

def empty_coauthor_detail_like():
    return pd.DataFrame(columns=["author_1_name","author_2_name","title","journal","year",
                                 "cited_by_count","doi","position_author_1","position_author_2"])

# ---- Validation ----
def ensure_loaded_author(author_id: str, df_authorship: pd.DataFrame):
    if author_id not in df_authorship["author_id"].values:
        raise ValueError("Author not loaded – run add_author() first.")

# ---- Year filters (handles missing years) ----
def filter_by_year(df: pd.DataFrame, col="year", year_from=None, year_to=None):
    out = df
    if year_from is not None:
        out = out[out[col].ge(year_from, fill_value=False)]
    if year_to is not None:
        out = out[out[col].le(year_to, fill_value=False)]
    return out

# ---- Safe sort with whitelist ----
def safe_sort(df: pd.DataFrame, by: str, allowed: set[str], ascending=False):
    if by not in allowed:
        raise ValueError(f"sort_by must be one of {allowed}")
    return df.sort_values(by, ascending=ascending, kind="mergesort")

# ---- Positions table for a given author (used by many functions) ----
def positions_for(author_id: str, authorship_df: pd.DataFrame, alias: str):
    return (authorship_df.loc[authorship_df["author_id"] == author_id, ["paper_id","position"]]
                        .drop_duplicates()
                        .rename(columns={"position": f"position_{alias}"}))


def upsert_author_row(dfs: DFs, author_id: str, a: dict):
    first, last = _split_name(a["display_name"])
    dfs.authors.loc[author_id] = {
        "first_name": first,
        "last_name":  last,
        "orcid":      a.get("orcid"),  # consider a.get("ids",{}).get("orcid") if schema differs
    }

def upsert_work_and_edges(dfs: DFs, work: dict, include_coauthors: bool):
    w_id   = work["id"].split("/")[-1]
    doi    = work.get("doi")
    pl_src = (work.get("primary_location") or {}).get("source") or {}
    journal = pl_src.get("display_name")

    dfs.papers.loc[w_id] = {
        "title":          work["display_name"],
        "journal":        journal,
        "year":           work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count", 0),
        "doi":            doi,
    }

    refs = work.get("referenced_works") or []
    if refs:
        rows = pd.DataFrame({"citing_paper": w_id,
                             "cited_paper": [r.split('/')[-1] for r in refs]})
        dfs.citations = pd.concat([dfs.citations, rows], ignore_index=True)

    for au in work.get("authorships", []):
        au_id = au["author"]["id"].split("/")[-1]
        pos   = au["author_position"]

        if include_coauthors and au_id not in dfs.authors.index:
            f, l = _split_name(au["author"]["display_name"])
            dfs.authors.loc[au_id] = {"first_name": f, "last_name": l, "orcid": au.get("orcid")}

        # append if not present
        dup = (dfs.authorship["author_id"] == au_id) & (dfs.authorship["paper_id"] == w_id)
        if not dup.any():
            dfs.authorship.loc[len(dfs.authorship)] = [au_id, w_id, pos]

def index_author(author_id: str, dfs: DFs, client: PyAlexClient, include_coauthors=True):
    a = client.fetch_author(author_id)
    upsert_author_row(dfs, author_id, a)
    for w in client.iter_works_by_author(author_id):
        upsert_work_and_edges(dfs, w, include_coauthors=include_coauthors)


def coauthored_papers(a1: str, a2: str, dfs: DFs,
                      *, year_from=None, year_to=None,
                      sort_by="cited_by_count", ascending=False) -> pd.DataFrame:
    if a1 == a2: raise ValueError("Provide two distinct author IDs.")
    present = set(dfs.authorship["author_id"])
    missing = [a for a in (a1, a2) if a not in present]
    if missing: raise ValueError(f"Author(s) not loaded: {missing}")

    p1 = positions_for(a1, dfs.authorship, "author_1")
    p2 = positions_for(a2, dfs.authorship, "author_2")
    both = p1.merge(p2, on="paper_id", how="inner")
    if both.empty: return empty_coauthored_like()

    out = (both.join(dfs.papers, on="paper_id")
               .set_index("paper_id")
               .loc[:, ["title","journal","year","cited_by_count","doi",
                        "position_author_1","position_author_2"]])

    out = filter_by_year(out, year_from=year_from, year_to=year_to)
    if out.empty: return empty_coauthored_like()
    return safe_sort(out, sort_by, {"cited_by_count","year","title"}, ascending)

def connections_between(author_ids: list[str],
                        dfs: DFs,
                        *, year_from: int | None = None,
                        year_to: int | None = None,
                        sort_by: str = "year",
                        ascending: bool = True) -> pd.DataFrame:
    """
    Build detailed coauthorship connections for all pairs in `author_ids`.
    Returns rows per shared paper with both authors' names & positions.
    """
    # narrow authorship to the provided IDs
    keep = dfs.authorship["author_id"].isin(set(author_ids))
    sub = dfs.authorship.loc[keep, ["author_id","paper_id","position"]].drop_duplicates()
    if sub.empty:
        return pd.DataFrame(columns=["paper_id","author_1_id","author_2_id","author_1_name","author_2_name",
                                     "title","journal","year","cited_by_count","doi",
                                     "position_author_1","position_author_2"])

    a = sub.rename(columns={"author_id":"author_1_id", "position":"position_author_1"})
    b = sub.rename(columns={"author_id":"author_2_id", "position":"position_author_2"})
    pairs = a.merge(b, on="paper_id", how="inner").query("author_1_id < author_2_id")
    if pairs.empty:
        return pd.DataFrame(columns=["paper_id","author_1_id","author_2_id","author_1_name","author_2_name",
                                     "title","journal","year","cited_by_count","doi",
                                     "position_author_1","position_author_2"])

    # join paper metadata
    meta_cols = ["title","journal","year","cited_by_count","doi"]
    out = pairs.join(dfs.papers[meta_cols], on="paper_id")

    # attach names
    names = (dfs.authors["first_name"].fillna("") + " " + dfs.authors["last_name"].fillna("")).rename("full_name")
    out = (out
           .join(names.rename("author_1_name"), on="author_1_id")
           .join(names.rename("author_2_name"), on="author_2_id"))

    # tidy columns + optional year filter + sort
    out = out.loc[:, ["paper_id","author_1_id","author_2_id","author_1_name","author_2_name",
                      "title","journal","year","cited_by_count","doi",
                      "position_author_1","position_author_2"]]
    out = filter_by_year(out, year_from=year_from, year_to=year_to)
    return out.sort_values(sort_by, ascending=ascending, kind="mergesort").reset_index(drop=True)

