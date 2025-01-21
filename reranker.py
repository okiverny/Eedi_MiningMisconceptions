
import vllm
import numpy as np
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from helpers import preprocess_text
from transformers import AutoTokenizer
from typing import List

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

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
    
    def _get_candidates(self, c_indices):
        candidates = []

        mis_names = self.misconception_df["MisconceptionName"].values
        for ix in c_indices:
            c_names = []
            for i, name in enumerate(mis_names[ix]):
                c_names.append(f"{i+1}. {name}")

            candidates.append("\n".join(c_names))
            
        return candidates
    
    def rerank_with_reasoning(self, n_iterations : int):

        llm = vllm.LLM(
            self.model_path,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.98, 
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True,
            max_model_len=2000,
            disable_log_stats=True,
            cpu_offload_gb=8,
            swap_space=1,
            device='cuda',
            max_num_seqs=20
        )
        self.tokenizer = llm.get_tokenizer()

        # Initialization of selected misconceptions (N, n_iterations)
        selected = np.full((self.indices.shape[0], n_iterations), -1, dtype=int)

        for iteration in range(n_iterations):
            print(f"Iteration {iteration + 1}/{n_iterations}")

            # Remove values from indices corresponding to survivors
            remaining = np.array([row[~np.isin(row, choices)] for row, choices in zip(self.indices, selected)])
            print('Removing llm choices from the first iteration. Now we have shape:', remaining.shape)

            # Split the array into the chunks of 9 + n*8 + rest
            chunks = self._split_array(remaining) if iteration<1 else self._split_array(remaining[:,:9])

            for ichunk, chunk in enumerate(chunks):
                print(f'Processing chunk {ichunk+1}/{len(chunks)}')

                # Add previous choices if available
                if chunk.shape[1]<9:
                    chunk = np.concatenate([chunk, best_values_local], axis=1)

                if ichunk==0:
                    chunk_choices = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
                else:
                    chunk_choices = [str(opt+1) for opt in range(chunk.shape[1])]
                #print('chunk_choices', chunk_choices)

                ### Block with LLM choice
                self.data["retrieval"] = self._get_candidates(chunk)
                self.data["text"] = self.data.apply(lambda row: self._apply_template(row, self.tokenizer), axis=1)

                #print("Example:")
                #print(df["text"].values[0])
                #print()

                # Run Qwen2.5 72b instruct LLM
                responses = llm.generate(
                    self.data["text"].values,
                    vllm.SamplingParams(
                        n=1,  # Number of output sequences to return for each prompt.
                        top_k=1,  # Float that controls the cumulative probability of the top tokens to consider.
                        temperature=0,  # randomness of the sampling
                        seed=777, # Seed for reprodicibility
                        skip_special_tokens=False,  # Whether to skip special tokens in the output.
                        max_tokens=1,  # Maximum number of tokens to generate per output sequence.
                        logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=chunk_choices)]
                    ),
                    use_tqdm=True
                )

                responses = [x.outputs[0].text for x in responses]
                self.data["response"] = responses

                best_indices_local = self.data["response"].astype(int).values - 1
                #print('llm_choices', best_indices_local)
                best_values_local = np.array([chunk_row[index_row] for index_row, chunk_row in zip(best_indices_local, chunk)]).reshape(-1, 1)

            print('Final LLM choices:')
            print(best_values_local)

            # New selected array
            selected[:, iteration] = best_values_local[:, 0]

        # Reranked array
        remaining = np.array([row[~np.isin(row, choices)] for row, choices in zip(self.indices, selected)])
        reranked = np.concatenate([selected, remaining], axis=1)

        # Return reranked array of results
        return reranked