import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc
import torch
from numpy.linalg import norm
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModel, BitsAndBytesConfig
import peft
from peft import (
    LoraConfig,
    get_peft_model,
)
from tqdm.autonotebook import trange
from abc import ABC, abstractmethod

from SearchMethods import RetrievalStrategy

class Qwen25_14B_Search(RetrievalStrategy):

    def _batch_to_device(self, batch, target_device):
        """ send a pytorch batch to a device (CPU/GPU) """
        for key in batch:
            if isinstance(batch[key], Tensor):
                batch[key] = batch[key].to(target_device)
        return batch
    
    def _last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        
    def _inference(self, df, model, tokenizer, device):
        batch_size = 16
        max_length = 512
        sentences = list(df['query_text'].values)

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=False):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True,
                             return_tensors="pt")
            features = self._batch_to_device(features, device)
            with torch.no_grad():
                outputs = model(**features)
                embeddings = self._last_token_pool(outputs.last_hidden_state, features['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                embeddings = embeddings.detach().cpu().numpy().tolist()
            all_embeddings.extend(embeddings)

        all_embeddings = [np.array(all_embeddings[idx]).reshape(1, -1) for idx in np.argsort(length_sorted_idx)]

        return np.concatenate(all_embeddings, axis=0)
    
    def _load_model_and_tokenizer(self, base_model_path, lora_path, load_in_4bit=True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModel.from_pretrained(base_model_path,
            quantization_config=bnb_config, 
            device_map='cuda:0',
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            lora_path if lora_path else base_model_path
        )
        model.resize_token_embeddings(len(tokenizer))
        if lora_path:
            model = peft.PeftModel.from_pretrained(model, lora_path)
        return model, tokenizer
    
    def _get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def _get_embeddings_in_batches(self, model, tokenizer, texts, max_length, batch_size=32):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i : i + batch_size]
            batch_dict = tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.no_grad(), torch.amp.autocast("cuda"):
                outputs = model(**batch_dict)
                batch_embeddings = self._last_token_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1).cpu()
            embeddings.append(batch_embeddings)
        return torch.cat(embeddings, dim=0)
    
    def _get_new_queries(self, queries, query_max_len, examples_prefix, tokenizer):
        inputs = tokenizer(
            queries,
            max_length=query_max_len - len(tokenizer('<s>', add_special_tokens=False)['input_ids']) - len(
                tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
            return_token_type_ids=False,
            truncation=True,
            return_tensors=None,
            add_special_tokens=False
        )
        prefix_ids = tokenizer(examples_prefix, add_special_tokens=False)['input_ids']
        suffix_ids = tokenizer('\n<response>', add_special_tokens=False)['input_ids']
        new_max_length = (len(prefix_ids) + len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
        new_queries = tokenizer.batch_decode(inputs['input_ids'])
        for i in range(len(new_queries)):
            new_queries[i] = examples_prefix + new_queries[i] + '\n<response>'
        return new_max_length, new_queries

    def search_misconceptions(self, data, misconception_mapping, top_k, model_path, lora_path):
        
        device='cuda:0'

        # Get the model
        model, tokenizer = self._load_model_and_tokenizer(model_path, lora_path, load_in_4bit=True)

        task_description = 'Given a math question with correct answer and a misconcepted incorrect answer, retrieve the most accurate misconception for the incorrect answer.'
        data['query_text'] = data['query_text'].apply(lambda x: self._get_detailed_instruct(task_description, x))

        queries = data['query_text'].to_list()
        documents = misconception_mapping['MisconceptionName'].tolist()

        query_max_len, doc_max_len = 512, 100
        examples_prefix = ''
        new_query_max_len, new_queries = self._get_new_queries(queries, query_max_len, examples_prefix, tokenizer)
        data = {'texts': new_queries+ documents}

        embeddings = self._get_embeddings_in_batches(
            model,
            tokenizer,
            data['texts'],
            max_length=512,
            batch_size=4,
        )
        text2embeds = {text: emb for text, emb in zip(data['texts'], embeddings)}

        query_embeddings = torch.stack([text2embeds[t] for t in new_queries])
        doc_embeddings = torch.stack([text2embeds[t] for t in documents])

        scores = query_embeddings @ doc_embeddings.T  # Shape: (M, N)
        sorted_indices = torch.argsort(scores,1, descending=True)[:,:top_k].tolist()

        np.save("indices.npy", np.array(sorted_indices))
        data.to_parquet("df.parquet", index=False)

        return data, sorted_indices