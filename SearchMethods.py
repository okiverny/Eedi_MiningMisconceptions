from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import time, tqdm, warnings
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string

# Strategy interface
class RetrievalStrategy(ABC):
    """This is the interface that declares operations common to all signal extraction algorithms/methods."""
    @abstractmethod
    def search_misconceptions(self, data: pd.DataFrame):
        pass

# Concrete Strategies
class LexicalSearch(RetrievalStrategy):
    def search_misconceptions(self, data: pd.DataFrame):
        return data
    
    