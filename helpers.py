import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction import _stop_words
import string

from eedi_metrics import mapk, apk

def preprocess_text(x):
    x = x.lower()                 # Convert words to lowercase
    x = re.sub("@\w+", '',x)      # Delete strings starting with @
    #x = re.sub("'\d+", '',x)      # Delete Numbers
    x = re.sub("http\w+", '',x)   # Delete URL
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = re.sub(r"\.+", ".", x)    # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = x.strip()                 # Remove empty characters at the beginning and end
    return x

def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

def combined_search(semantic_results, semantic_scores, keyword_results, keyword_scores, misconception_mapping, top_k=25, alpha=0.95):
    combined_results = []
    for sem_res, sem_scores, key_res, key_scores in zip(semantic_results, semantic_scores, keyword_results, keyword_scores):

        # Normalize semantic scores (now smaller is better)
        sem_scores_norm = (sem_scores - np.min(sem_scores)) / (np.max(sem_scores) - np.min(sem_scores))

        # Normalize keyword scores
        key_scores_norm = (key_scores - np.min(key_scores)) / (np.max(key_scores) - np.min(key_scores))

        # Compute combined scores with the alpha and 1-alpha weights
        combined_scores = np.zeros(len(misconception_mapping))
        for idx, score in zip(sem_res, sem_scores_norm):
            combined_scores[idx] += alpha * score

        for idx, score in zip(key_res, key_scores_norm):
            combined_scores[idx] += (1 - alpha) * score

        top_combined = np.argsort(combined_scores)[::-1][:top_k]
        combined_results.append(top_combined)

    return combined_results

def evaluate(data: pd.DataFrame, preds_col: str, misconception_id_txt: dict) -> pd.DataFrame:
    # Compute APK score for the BM25 search results
    data['score'] = data.apply(lambda row: apk([row['MisconceptionId']], row[preds_col]), axis=1)
    data['score10'] = data.apply(lambda row: apk([row['MisconceptionId']], row[preds_col], k=10), axis=1)
    data['score1'] = data.apply(lambda row: apk([row['MisconceptionId']], row[preds_col], k=1), axis=1)


    # Print outs for examination
    for idx, row in data.iterrows():
        score = apk([row['MisconceptionId']], row[preds_col])
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

    print(f'MAPK@25 (BM25) = {np.mean(data.score):.4f}')
    print(f'MAPK@10 (BM25) = {np.mean(data.score10):.4f}')
    print(f'MAPK@1 (BM25) = {np.mean(data.score1):.4f}')

    return data