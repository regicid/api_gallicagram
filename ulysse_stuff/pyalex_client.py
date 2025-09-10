from pyalex import Authors, Works
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
    from toolbox import init_tables, DFs, index_author
    from custom_index import h_index_overall, h_index_first, h_index_middle, h_index_last
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
