import os
import numpy as np
import pandas as pd
import time

from DataExtractor import process_question, standardize_question_data, process_data
from Retrieval import MisconceptRetrieval
from SearchMethods import LexicalSearch
from eedi_metrics import mapk, apk

if __name__ == "__main__":
    # Reading data
    df_train = pd.read_csv("data/train.csv").set_index('QuestionId')
    df_test = pd.read_csv("data/test.csv").set_index('QuestionId')

    # Misconceptions: create dictionary for fast substitution
    misconception_mapping = pd.read_csv("data/misconception_mapping.csv").set_index('MisconceptionId')
    misconception_id_txt = dict(zip(misconception_mapping.index, misconception_mapping.MisconceptionName))
    misconception_txt_id = {miscName: miscIndex for miscIndex, miscName in misconception_id_txt.items()}

    #start = time.time()

    is_labeled = True
    data = process_data(df_train, is_labeled)
    # Filter NaNs
    data = data[data['MisconceptionId'].notnull()]
    data['MisconceptionId'] = data['MisconceptionId'].astype('Int32')
    data = data.head(10)

    #end = time.time()
    #print(f"time: {end - start:.4f} seconds")

    # Analysis
    #print(706, misconception_mapping.loc[[706]].MisconceptionName.values[0])
    #print(df_test.ConstructName.values[0])

    print('Number of Unique misconceptions in train data:', len(data.MisconceptionId.unique()))
    print('Number of All known misconceptions:',len(misconception_mapping))

    #print(misconception_mapping.MisconceptionName)
    #print(df_train.head(10).QuestionText)

    misconceptions = misconception_mapping.MisconceptionName.values
    queries = data.QuestionText + '_' + data.ConstructName #+ '_' + data.IncorrectAnswer

    # Initialize the MisconceptRetrieval with a lexical search strategy
    retrieval = MisconceptRetrieval(LexicalSearch())
    top_results, scores = retrieval.find_misconceptions(misconceptions, queries, 25)

    top_results_index = []
    for i in range(len(top_results)):
        top_results_index.append([misconception_txt_id[miscName] for miscName in top_results[i]])

    # Submission format
    data['predictions'] = top_results_index
    #print(data['MisconceptionId'])
    data['score'] = data.apply(lambda row: apk([row['MisconceptionId']], row['predictions']), axis=1)

    for idx, row in data.iterrows():
        print(row['MisconceptionId'], row['predictions'])
        score = apk([row['MisconceptionId']], row['predictions'])
    

    
    print(data)
    print(f'MAPK@25 (BM25) = {np.mean(data.score):.4f}')

