import os
import numpy as np
import pandas as pd
import time

from DataExtractor import process_data
from Retrieval import MisconceptRetrieval
from SearchMethods import LexicalSearch, SemanticSearch, SemanticSearchReranking
from eedi_metrics import mapk, apk
from helpers import combined_search

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
    queries =  data.SubjectName + ' ' + data.ConstructName + '. Question ' + data.QuestionText + '. Answer ' + data.IncorrectAnswer +'.'
    #queries =  '{ topic: ' + data.SubjectName + ', detailed topic: ' + data.ConstructName + '.\n, question: ' + data.QuestionText.replace("\n", " ") + '.\n, incorrect answer: ' + data.IncorrectAnswer.replace("\n", " ") +'.}'

    # Initialize the MisconceptRetrieval with a lexical search strategy
    retrieval = MisconceptRetrieval(LexicalSearch())
    BM25_results, BM25_scores = retrieval.find_misconceptions(misconceptions, queries, 25)
    data['BM25preds'] = [[misconception_txt_id[miscName] for miscName in BM25_results[i]] for i in range(len(BM25_results))]
    data['BM25scores'] = list(BM25_scores)

    ############
    retrieval = MisconceptRetrieval(SemanticSearch())
    semantic_results, semantic_scores = retrieval.find_misconceptions(misconceptions, queries, 25)
    data['sem_preds'] = [[misconception_txt_id[miscName] for miscName in semantic_results[i]] for i in range(len(semantic_results))]
    data['sem_scores'] = list(semantic_scores)
    ###########

    # Combine Lexical and Semantic search results
    hybrid_results = combined_search(data['sem_preds'], data['sem_scores'], data['BM25preds'], data['BM25scores'], misconception_mapping)
    data['hybrid_preds'] = hybrid_results

    #retrieval = MisconceptRetrieval(SemanticSearchReranking())
    #BM25_results, BM25_scores = retrieval.find_misconceptions(misconceptions, queries, 25)



    preds_col = 'hybrid_preds'

    # Compute APK score for the BM25 search results
    data['score'] = data.apply(lambda row: apk([row['MisconceptionId']], row[preds_col]), axis=1)
    data['score10'] = data.apply(lambda row: apk([row['MisconceptionId']], row[preds_col], k=10), axis=1)
    data['score1'] = data.apply(lambda row: apk([row['MisconceptionId']], row[preds_col], k=1), axis=1)


    # Print outs for examination
    for idx, row in data.iterrows():
        score = apk([row['MisconceptionId']], row[preds_col])
        score10 = apk([row['MisconceptionId']], row[preds_col], k=10)
        score1 = apk([row['MisconceptionId']], row[preds_col], k=1)
        print()
        print(row['MisconceptionId'], '---', row[preds_col], score)
        print('      ', row['BM25scores'])
        if score == 0.0:
            print('----->', row.SubjectName)
            print('----->', row.ConstructName)
            print(row.QuestionText)
            print('Correct Answer:', row.CorrectAnswerText)
            print('Incorrect Answer:', row.IncorrectAnswer)
            print('Correct misconception: ', misconception_id_txt[row['MisconceptionId']])
            print('First rank misconception: ', misconception_id_txt[row[preds_col][0]])
            print(30*'=')
    
    end = time.time()
    print(f"time: {end - start:.4f} seconds")

    
    print(data)
    print(f'MAPK@25 (BM25) = {np.mean(data.score):.4f}')
    print(f'MAPK@10 (BM25) = {np.mean(data.score10):.4f}')
    print(f'MAPK@1 (BM25) = {np.mean(data.score1):.4f}')

