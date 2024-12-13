import os
import numpy as np
import pandas as pd
import time

from DataExtractor import process_data
from Retrieval import MisconceptRetrieval
from SearchMethods import LexicalSearch, SemanticSearch, SemanticSearchFineTuned, SemanticSearchPadEmbedding
from helpers import combined_search, evaluate, reciprocal_rank_fusion, combined_search_three_models

def run_cpu(input_data: pd.DataFrame, is_labeled: bool, misconception_mapping: pd.DataFrame) -> pd.DataFrame:
    start = time.time()

    # Misconceptions: create dictionary for fast substitution
    misconception_id_txt = dict(zip(misconception_mapping.index, misconception_mapping.MisconceptionName))
    misconception_txt_id = {miscName: miscIndex for miscIndex, miscName in misconception_id_txt.items()}
    misconceptions = misconception_mapping.MisconceptionName.values

    # process input data
    data = process_data(input_data, is_labeled)

    # Filter NaNs
    data = data[data['MisconceptionId'].notnull()]
    data['MisconceptionId'] = data['MisconceptionId'].astype('Int32')

    print('Number of Unique misconceptions in train data:', len(data.MisconceptionId.unique()))
    print('Number of All known misconceptions:', len(misconception_mapping))

    # Query for BM25 search
    queries_lexic =  data.SubjectName + ' ' + data.ConstructName + '. Question ' + data.QuestionText + '. Answer ' + data.IncorrectAnswer +'.'

    ### Query for fine-tuned models such as BGE
    queries =  data.SubjectName + ' (' + data.ConstructName + ').\nQuestion: ' + data.QuestionText + '\nCorrect Answer: ' + data.CorrectAnswerText +'\nIncorrect Answer: ' + data.IncorrectAnswer

    ###### Initialize the MisconceptRetrieval with a lexical search strategy
    retrieval = MisconceptRetrieval(LexicalSearch())
    BM25_results, BM25_scores = retrieval.find_misconceptions(misconceptions, queries_lexic, 25)
    data['BM25preds'] = [[misconception_txt_id[miscName] for miscName in BM25_results[i]] for i in range(len(BM25_results))]
    data['BM25scores'] = list(BM25_scores)

    ############ Semantic Search with the base ms-marco-MiniLM-L-6-v2 (alpha=0.7)
    retrieval = MisconceptRetrieval(SemanticSearch())
    semantic_results, semantic_scores = retrieval.find_misconceptions(misconceptions, queries_lexic, 25)
    data['sem_preds'] = [[misconception_txt_id[miscName] for miscName in semantic_results[i]] for i in range(len(semantic_results))]
    data['sem_scores'] = list(semantic_scores)

    hybrid_results, hybrid_scores = combined_search(data['sem_preds'], data['sem_scores'], data['BM25preds'], data['BM25scores'], misconception_mapping, top_k=25, alpha=0.70)
    data['comb_preds'] = hybrid_results
    data['comb_scores'] = list(hybrid_scores)

    ########### Semantic Search (GTE Fine-Tuned) (alpha=0.85)
    retrieval = MisconceptRetrieval(SemanticSearchFineTuned())
    model_path = '/Users/okiverny/workspace/Kaggle/Eedi_MiningMisconceptions/models/gte-base-weights/gte-base_trained_model_version2'
    semantic_results, semantic_scores = retrieval.find_misconceptions(misconceptions, queries, 25, model_path)
    data['gte_preds'] = [[misconception_txt_id[miscName] for miscName in semantic_results[i]] for i in range(len(semantic_results))]
    data['gte_scores'] = list(semantic_scores)

    hybrid_results, hybrid_scores = combined_search(data['gte_preds'], data['gte_scores'], data['BM25preds'], data['BM25scores'], misconception_mapping, top_k=25, alpha=0.85)
    data['gte_comb_preds'] = hybrid_results
    data['gte_comb_scores'] = list(hybrid_scores)

    ########### Semantic Search (mpnetV2 Fine-Tuned) (alpha=0.85)
    retrieval = MisconceptRetrieval(SemanticSearchFineTuned())
    model_path = '/Users/okiverny/workspace/Kaggle/Eedi_MiningMisconceptions/models/mpnet_weights_version1/mpnetV2_trained_model_version3'
    semantic_results, semantic_scores = retrieval.find_misconceptions(misconceptions, queries, 25, model_path)
    data['mpnet_preds'] = [[misconception_txt_id[miscName] for miscName in semantic_results[i]] for i in range(len(semantic_results))]
    data['mpnet_scores'] = list(semantic_scores)

    hybrid_results, hybrid_scores = combined_search(data['mpnet_preds'], data['mpnet_scores'], data['BM25preds'], data['BM25scores'], misconception_mapping, top_k=25, alpha=0.85)
    data['mpnet_comb_preds'] = hybrid_results
    data['mpnet_comb_scores'] = list(hybrid_scores)

    # mpnet+gte comb
    hybrid_results, hybrid_scores = combined_search(data['gte_preds'], data['gte_scores'], data['mpnet_preds'], data['mpnet_scores'], misconception_mapping, top_k=25, alpha=0.70)
    data['hybrid_preds'] = hybrid_results
    data['hybrid_scores'] = list(hybrid_scores)

    # Final retrieval
    data['hybrid_preds'] = hybrid_results

    # Compute MAP scores if possible
    if is_labeled:
        preds_col = 'hybrid_preds'
        data = evaluate(data, preds_col, misconception_id_txt)

    # Measuring the running time
    end = time.time()
    print(f"time: {end - start:.4f} seconds")

    return data

if __name__ == "__main__":

    RUNNING_MODE = 'cpu' # two versions of the code to run on CPU only (Efficiency Prize) or on GPU

    # Reading data
    df_train = pd.read_csv("data/train.csv").set_index('QuestionId')
    df_test = pd.read_csv("data/test.csv").set_index('QuestionId')
    misconception_mapping = pd.read_csv("data/misconception_mapping.csv").set_index('MisconceptionId')


    if RUNNING_MODE=='cpu':
        is_labeled = True
        data = run_cpu(df_train, is_labeled, misconception_mapping)
