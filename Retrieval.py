import pandas as pd

from SearchMethods import RetrievalStrategy, LexicalSearch

class MisconceptRetrieval:
    """The MisconceptRetrieval defines the interface to SearchMethods."""

    def __init__(self, search_strategy: RetrievalStrategy) -> None:
        self._search_strategy = search_strategy

    def background_strategy(self) -> RetrievalStrategy:
        """Just in case we need a reference to one of RetrievalStrategy objects"""
        return self._search_strategy

    def set_background_strategy(self, search_strategy: RetrievalStrategy) -> None:
        """Usually, the MisconceptRetrieval allows replacing a RetrievalStrategy object at runtime."""
        self._search_strategy = search_strategy

    def find_misconceptions(self, data):

        return 0