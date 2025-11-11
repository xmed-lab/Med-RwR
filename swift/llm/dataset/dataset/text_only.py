import ast
import os
import re
from typing import Any, Dict, Optional

import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from tqdm import tqdm

from ..media import MediaResource
from ..preprocessor import GroundingMixin, MessagesPreprocessor, ResponsePreprocessor, RowPreprocessor
from ..register import DatasetMeta, SubsetDataset, register_dataset

import pdb


class MedMCQARetrievePreprocessor(RowPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = """You are an experienced expert in medicine. You are given a question and a list of choices. You are required to select the correct answer from the choices.
First think about the question and each choice within <think> </think> tags. During thinking, if needed, retrieve medical knowledge using <query> </query> tags. Only one query is allowed. An external agent will retrieve information and return it within <retrieve> </retrieve> tags. 
You can use the retrieved information to continue thinking and further query if more information is needed. When you can reach a conclusion, output your answer within <answer> </answer> tags.
The output should be in the following format:
1. If you need more information, output <think> ... </think>\n<query> ... </query>\n<retrieve> ... </retrieve>\n (Multiple think-query-retrieve cycles may occur)
2. If you can directly reach a conclusion without query, output <think> ... </think>\n<answer> ... </answer>"""
        
        cop2answer = ["A", "B", "C", "D"]
        
        question_id = row["id"]
        question = row["question"]
        candidates = "\n".join([f"A. {row['opa']}", f"B. {row['opb']}", f"C. {row['opc']}", f"D. {row['opd']}"])
        answer = cop2answer[row["cop"]]
        
        return {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": question + "\n" + candidates
                }, {
                    "role": "assistant",
                    "content": answer
                }
            ],
            "answer": answer,
            "question_id": question_id,
            'images': []
        }

class MedQAUSMLEPreprocessorRetrieve(RowPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = """You are an experienced expert in medicine. You are given a question and a list of choices. You are required to select the correct answer from the choices.
First think about the question and each choice within <think> </think> tags. During thinking, if needed, retrieve medical knowledge using <query> </query> tags. Only one query is allowed. An external agent will retrieve information and return it within <retrieve> </retrieve> tags. 
You can use the retrieved information to continue thinking and further query if more information is needed. When you can reach a conclusion, output your answer within <answer> </answer> tags.
The output should be in the following format:
1. If you need more information, output <think> ... </think>\n<query> ... </query>\n<retrieve> ... </retrieve>\n (Multiple think-query-retrieve cycles may occur)
2. If you can directly reach a conclusion without query, output <think> ... </think>\n<answer> ... </answer>"""
 
        question = row["question"]
        options = row["options"]
        answer = row["answer"]

        for k, v in options.items():
            if v == answer:
                answer_idx = k
                break

        candidates = "\n".join([f"A. {options['A']}", f"B. {options['B']}", f"C. {options['C']}", f"D. {options['D']}"])
        
        return {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": question + "\n" + candidates
                }, {
                    "role": "assistant",
                    "content": answer_idx
                }
            ],
            "answer": answer,
            'images': []
        }


register_dataset(
    DatasetMeta(
        dataset_path="",
        preprocess_func=MedQAUSMLEPreprocessorRetrieve(),
        tags=['long-sequence', 'QA']))


register_dataset(
    DatasetMeta(
        dataset_path="",
        preprocess_func=MedQAUSMLEPreprocessorRetrieve(),
        tags=['long-sequence', 'QA']))
