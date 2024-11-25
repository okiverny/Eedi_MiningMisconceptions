import os
import numpy as np
import pandas as pd
import time

from DataExtractor import process_question, standardize_question_data, process_data
from Retrieval import MisconceptRetrieval
from SearchMethods import LexicalSearch

if __name__ == "__main__":
    # Reading data
    df_train = pd.read_csv("data/train.csv").set_index('QuestionId')
    df_test = pd.read_csv("data/test.csv").set_index('QuestionId')
    misconception_mapping = pd.read_csv("data/misconception_mapping.csv").set_index('MisconceptionId')

    start = time.time()

    is_labeled = True
    data = process_data(df_train, is_labeled)
    print(data.head(5))

    end = time.time()
    print(f"time: {end - start:.4f} seconds")

    # Analysis
    print(706, misconception_mapping.loc[[706]].MisconceptionName.values[0])

    print(df_test.ConstructName.values[0])

    print('Number of Unique misconceptions in train data:', len(data.MisconceptionId.unique()))
    print('Number of All known misconceptions:',len(misconception_mapping))


    
    # Initialize the MisconceptRetrieval with a lexical search strategy
    retrieval = MisconceptRetrieval(LexicalSearch())