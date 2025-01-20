
import vllm
import numpy as np
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from helpers import preprocess_text

class RerankerModel:
    def __init__(self, model_path: str, indices_path: str = "indices.npy", data_path : str = "df.parquet"):
        self.model_path = model_path
        self.indices = np.load(indices_path)
        self.data = pd.read_parquet(data_path)
        self.misconception_df = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
        self.PROMPT = """You are a Mathematics teacher. Your task is to identify and explain the misconception that leads to an incorrect answer in the given calculation problem. This diagnostic question measures the student’s ability to {ConstructName} ({SubjectName}).

                    Here is the problem statement:
                    Question: {Question}
                    Correct Answer: {CorrectAnswer}
                    Incorrect Answer: {IncorrectAnswer}

                    Identify the misconception behind the Incorrect Answer from the list below. Provide only the correct misconception number.

                    Misconception options:
                    {Retrival}
                    """

    def _split_array(self, data):
        """
        Splits the given array into chunks:
        - The first chunk has the last 9 columns (N, 9).
        - The remaining chunks each have 8 columns (N, 8).
        - The last chunk may contain fewer than 8 columns if there are leftovers.
        
        Parameters:
        - data (numpy.ndarray): Array of shape (N, K) to be split.
        
        Returns:
        - list: A list of numpy arrays, each being a chunk of shape (N, 9), (N, 8), or (N, X) for the final chunk.
        """
        N, K = data.shape
        chunks = []
        
        # First chunk: last 9 columns
        if K<=9:
            chunks.append(data)
            return chunks
        else:
            chunks.append(data[:, -9:])

        if (K-9) // 8 > 0:
            for i in range((K-9) // 8):
                chunks.append(data[:, -9-8*(i+1):-9-8*i])
        
        # Remaining chunk (if any)
        if (K-9) % 8 != 0:
            chunks.append(data[:, 0:K - 9 - 8 * (len(chunks)-1)])
        
        return chunks
    
    def _apply_template(self, row, tokenizer):
        messages = [
            {
                "role": "user", 
                "content": preprocess_text(
                    self.PROMPT.format(
                        ConstructName=row["ConstructName"],
                        SubjectName=row["SubjectName"],
                        Question=row["QuestionText"],
                        IncorrectAnswer=row["incorrect_answer"],
                        CorrectAnswer=row["correct_answer"],
                        Retrival=row["retrieval"]
                    )
                )
            }
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text
