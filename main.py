import os
import numpy as np
import pandas as pd
import time

from DataExtractor import process_data
from Retrieval import MisconceptRetrieval
from SearchMethods import LexicalSearch, SemanticSearch, SemanticSearchFineTuned
from helpers import combined_search, evaluate

if __name__ == "__main__":
    # Reading data
    df_train = pd.read_csv("data/train.csv").set_index('QuestionId')
    df_test = pd.read_csv("data/test.csv").set_index('QuestionId')

    # Misconceptions: create dictionary for fast substitution
    misconception_mapping = pd.read_csv("data/misconception_mapping.csv").set_index('MisconceptionId')
    misconception_id_txt = dict(zip(misconception_mapping.index, misconception_mapping.MisconceptionName))
    misconception_txt_id = {miscName: miscIndex for miscIndex, miscName in misconception_id_txt.items()}


    is_labeled = True
    data = process_data(df_train, is_labeled)
    # Filter NaNs
    data = data[data['MisconceptionId'].notnull()]
    data['MisconceptionId'] = data['MisconceptionId'].astype('Int32')

    print('Number of Unique misconceptions in train data:', len(data.MisconceptionId.unique()))
    print('Number of All known misconceptions:',len(misconception_mapping))

    #print(misconception_mapping.MisconceptionName)
    #print(df_train.head(10).QuestionText)

    start = time.time()

    misconceptions = misconception_mapping.MisconceptionName.values
    queries_lexic =  data.SubjectName + ' ' + data.ConstructName + '. Question ' + data.QuestionText + '. Answer ' + data.IncorrectAnswer +'.'
    #queries =  '{ topic: ' + data.SubjectName + ', detailed topic: ' + data.ConstructName + '.\n, question: ' + data.QuestionText.replace("\n", " ") + '.\n, incorrect answer: ' + data.IncorrectAnswer.replace("\n", " ") +'.}'
    ### For BGE fine-tuned
    queries =  data.SubjectName + ' (' + data.ConstructName + ').\nQuestion: ' + data.QuestionText + '\nCorrect Answer: ' + data.CorrectAnswerText +'\nIncorrect Answer: ' + data.IncorrectAnswer


    ###### Initialize the MisconceptRetrieval with a lexical search strategy
    retrieval = MisconceptRetrieval(LexicalSearch())
    BM25_results, BM25_scores = retrieval.find_misconceptions(misconceptions, queries_lexic, 25)
    data['BM25preds'] = [[misconception_txt_id[miscName] for miscName in BM25_results[i]] for i in range(len(BM25_results))]
    data['BM25scores'] = list(BM25_scores)

    ############ Semantic Search
    # retrieval = MisconceptRetrieval(SemanticSearch())
    # semantic_results, semantic_scores = retrieval.find_misconceptions(misconceptions, queries, 25)
    # data['sem_preds'] = [[misconception_txt_id[miscName] for miscName in semantic_results[i]] for i in range(len(semantic_results))]
    # data['sem_scores'] = list(semantic_scores)

    ########### Semantic Search (Fine-Tuned)
    retrieval = MisconceptRetrieval(SemanticSearchFineTuned())
    semantic_results, semantic_scores = retrieval.find_misconceptions(misconceptions, queries, 25)
    data['sem_preds'] = [[misconception_txt_id[miscName] for miscName in semantic_results[i]] for i in range(len(semantic_results))]
    data['sem_scores'] = list(semantic_scores)

    # Combine Lexical and Semantic search results
    hybrid_results = combined_search(data['sem_preds'], data['sem_scores'], data['BM25preds'], data['BM25scores'], misconception_mapping)
    data['hybrid_preds'] = hybrid_results


    preds_col = 'hybrid_preds'
    #preds_col = 'sem_preds'
    data = evaluate(data, preds_col, misconception_id_txt)
    
    end = time.time()
    print(f"time: {end - start:.4f} seconds")

    #print(data)

