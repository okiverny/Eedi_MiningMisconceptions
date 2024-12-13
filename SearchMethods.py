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
    def search_misconceptions(self, texts, queries, top_k, model_path=''):
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

            
            for top_i, hit in enumerate(bm25_hits[0:top_k]):
                scores[iquery][top_i] = hit['score']
                results[iquery][top_i] = texts[hit['corpus_id']]

        
        return results, scores

class SemanticSearch(RetrievalStrategy):
    #### Semantic Search (bi-encoder)
    def search_misconceptions(self, texts, queries, top_k, model_path=''):

        bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device='cpu') # MAPK@25 = 0.1845 device='cpu' or `mps`
        #bi_encoder = SentenceTransformer('multi-qa-mpnet-base-cos-v1') # MAPK@25 = 0.1851
        #bi_encoder = SentenceTransformer('all-mpnet-base-v2')  # MAPK@25 = 0.1841
        #bi_encoder = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True) # MAPK@25 = 0.1730
        bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
        print('Encoding misconceptions ...')
        misconception_embeddings = bi_encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)

        print('Encoding questions ...')
        query_embeddings = bi_encoder.encode(queries.values, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
        hits = util.semantic_search(query_embeddings, misconception_embeddings, top_k=top_k, score_function=util.dot_score)

        # get top results and corresponding scores
        scores = np.zeros((len(queries), top_k))
        results = [[0 for _ in range(top_k)] for _ in range(len(queries))]

        for iquery, iquery_hits in enumerate(hits):
            for top_i, hit in enumerate(iquery_hits):
                scores[iquery][top_i] = hit['score']
                results[iquery][top_i] = texts[hit['corpus_id']]

        return results, scores

class SemanticSearchReranking(RetrievalStrategy):
    #### Semantic Search (bi-encoder + cross-encode)
    def search_misconceptions(self, texts, queries, top_k, model_path=''):

        bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
        print('Encoding misconceptions ...')
        misconception_embeddings = bi_encoder.encode(texts, convert_to_tensor=True, show_progress_bar=True)

        print('Encoding questions ...')
        query_embeddings = bi_encoder.encode(queries.values, convert_to_tensor=True, show_progress_bar=True)
        hits = util.semantic_search(query_embeddings, misconception_embeddings, top_k=top_k)

        ##### Re-Ranking #####
        #The bi-encoder retrieved top_k misconceptions. We use a cross-encoder, to re-rank the results list to improve the quality
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        new_hits = []
        for iquery, query_hits in enumerate(tqdm(hits)):
            query = queries.values[iquery]
            cross_inp = [[query, texts[hit['corpus_id']]] for hit in query_hits]
            cross_scores = cross_encoder.predict(cross_inp)

            # Sort results by the cross-encoder scores
            for idx in range(len(cross_scores)):
                query_hits[idx]['cross-score'] = cross_scores[idx]

            query_hits = sorted(query_hits, key=lambda x: x['cross-score'], reverse=True)
            new_hits.append(query_hits)


        # get top results and corresponding scores
        scores = np.zeros((len(queries), top_k))
        cross_scores = np.zeros((len(queries), top_k))
        results = [[0 for _ in range(top_k)] for _ in range(len(queries))]

        for iquery, iquery_hits in enumerate(new_hits):
            for top_i, hit in enumerate(iquery_hits):
                scores[iquery][top_i] = hit['score']
                cross_scores[iquery][top_i] = hit['cross-score']
                results[iquery][top_i] = texts[hit['corpus_id']]

        return results, cross_scores

class SemanticSearchFineTuned(RetrievalStrategy):
    #### Semantic Search on fine-tuned model (bi-encoder)
    def search_misconceptions(self, texts, queries, top_k, model_path=''):

        if model_path=='':
            #bi_encoder = SentenceTransformer('/Users/okiverny/workspace/Kaggle/Eedi_MiningMisconceptions/models/Eedi-finetuned-bge', device='mps') # device='cpu'
            bi_encoder = SentenceTransformer('/Users/okiverny/workspace/Kaggle/Eedi_MiningMisconceptions/models/gte-base-weights/gte-base_trained_model_version2', device='cpu', trust_remote_code=True, local_files_only=True)
            #bi_encoder = SentenceTransformer('/Users/okiverny/workspace/Kaggle/Eedi_MiningMisconceptions/models/mpnet_weights_version1/mpnetV2_trained_model_version3', device='mps', trust_remote_code=True, local_files_only=True)
        else:
            bi_encoder = SentenceTransformer(model_path, device='cpu', trust_remote_code=True, local_files_only=True)

        bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
        print('Encoding misconceptions ...')
        misconception_embeddings = bi_encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)

        print('Encoding questions ...')
        #queries = queries.apply(lambda x: preprocess_text(x))
        queries = queries.apply(preprocess_text)
        query_embeddings = bi_encoder.encode(queries.values, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)

        # Perform the semantic search
        hits = util.semantic_search(query_embeddings, misconception_embeddings, top_k=top_k, score_function=util.dot_score)

        # get top results and corresponding scores
        scores = np.zeros((len(queries), top_k))
        results = [[0 for _ in range(top_k)] for _ in range(len(queries))]

        for iquery, iquery_hits in enumerate(hits):
            for top_i, hit in enumerate(iquery_hits):
                scores[iquery][top_i] = hit['score']
                results[iquery][top_i] = texts[hit['corpus_id']]

        return results, scores