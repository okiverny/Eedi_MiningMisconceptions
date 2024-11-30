from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import time, warnings
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from helpers import bm25_tokenizer, preprocess_text


# Strategy interface
class RetrievalStrategy(ABC):
    """This is the interface that declares operations common to all signal extraction algorithms/methods."""
    @abstractmethod
    def search_misconceptions(self, texts, queries, top_k):
        pass

# Concrete Strategies
class LexicalSearch(RetrievalStrategy):
    #### BM25 Search (lexical search)
    def search_misconceptions(self, texts, queries, top_k):
        num_candidates=100
        #print('texts', texts)

        confusing_words = ['recall']

        tokenized_texts = []
        for passage in tqdm(texts):
            tokenized_texts.append(bm25_tokenizer(passage))

        bm25 = BM25Okapi(tokenized_texts)

        scores = np.zeros((len(queries), top_k))
        results = [[0 for _ in range(top_k)] for _ in range(len(queries))]
        # Processing queries
        for iquery, query in enumerate(queries):
            query_tokenized = [tok for tok in bm25_tokenizer(query) if tok not in confusing_words]
            bm25_scores = bm25.get_scores(query_tokenized)
            top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
            bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
            bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

            #print('------> Query:\n', query )
            #print('------> Query Tokens:\n', query_tokenized )
            #print(f"Top-{top_k} lexical search (BM25) hits of the following query:")
            
            for top_i, hit in enumerate(bm25_hits[0:top_k]):
                #print("\t{:.3f}\t{}".format(hit['score'], texts[hit['corpus_id']].replace("\n", " ")))
                scores[iquery][top_i] = hit['score']
                results[iquery][top_i] = texts[hit['corpus_id']]

        
        return results, scores

class SemanticSearch(RetrievalStrategy):
    #### Semantic Search (bi-encoder)
    def search_misconceptions(self, texts, queries, top_k):

        bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1') # MAPK@25 = 0.1845
        #bi_encoder = SentenceTransformer('multi-qa-mpnet-base-cos-v1') # MAPK@25 = 0.1851
        #bi_encoder = SentenceTransformer('all-mpnet-base-v2')  # MAPK@25 = 0.1841
        #bi_encoder = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True) # MAPK@25 = 0.1730
        bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
        print('Encoding misconceptions ...')
        misconception_embeddings = bi_encoder.encode(texts, convert_to_tensor=True, show_progress_bar=True)

        print('Encoding questions ...')
        query_embeddings = bi_encoder.encode(queries.values, convert_to_tensor=True, show_progress_bar=True)
        hits = util.semantic_search(query_embeddings, misconception_embeddings, top_k=top_k)

        # get top results and corresponding scores
        scores = np.zeros((len(queries), top_k))
        results = [[0 for _ in range(top_k)] for _ in range(len(queries))]

        for iquery, iquery_hits in enumerate(hits):
            for top_i, hit in enumerate(iquery_hits):
                scores[iquery][top_i] = hit['score']
                results[iquery][top_i] = texts[hit['corpus_id']]

        return results, scores

    
    
