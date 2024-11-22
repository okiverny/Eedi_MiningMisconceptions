import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from DataExtractor import process_question

# Model choice MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Reading data
df_train = pd.read_csv("data/train.csv").set_index('QuestionId')
df_test = pd.read_csv("data/test.csv").set_index('QuestionId')
misconception_mapping = pd.read_csv("data/misconception_mapping.csv").set_index('MisconceptionId')

# Model
#model = SentenceTransformer("all-MiniLM-L6-v2")
#print(model)

# Reading question block
question_id = 1869
result = process_question(df_test, question_id)
print(result)


# Analysis
print(706, misconception_mapping.loc[[706]].MisconceptionName.values[0])

print(df_test.ConstructName.values[0])