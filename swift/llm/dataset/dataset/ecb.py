import ast
import os
from typing import Any, Dict, Optional

import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from tqdm import tqdm

from ..media import MediaResource
from ..preprocessor import GroundingMixin, MessagesPreprocessor, ResponsePreprocessor, RowPreprocessor
from ..register import DatasetMeta, SubsetDataset, register_dataset

import pdb

class ECBenchPreprocessor(RowPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        image = [os.path.join("", img) for img in row["images"]]
        question = row["question"]
        answer = row["answer"]

        if row.get("similar_image", None) is not None:
            image.append(os.path.join("", row["similar_image"]))

        system_prompt = """You are an experienced expert in medicine. You are given a question, an image and a list of choices. You are required to select the correct answer from the choices.
First observe the image, think about the question and each choice within <think> </think> tags. During thinking, if needed, retrieve medical knowledge using <query> </query> tags. Only one query is allowed. An external agent will retrieve information and return it within <retrieve> </retrieve> tags. 
You can use the retrieved information to continue thinking and further query if more information is needed. When you can reach a conclusion, output your answer within <answer> </answer> tags.
The output should be in the following format:
1. If you need more information, output <think> ... </think>\n<query> ... </query>\n<retrieve> ... </retrieve>\n (Multiple think-query-retrieve cycles may occur)
2. If you can directly reach a conclusion without query, output <think> ... </think>\n<answer> ... </answer>"""
 
    
        return {
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt
                }, {
                    'role': 'user',
                    'content': question
            }, {
                'role': 'assistant',
                'content': answer
            }],
            'images': image,
            "answer": answer,
        }
        

register_dataset(
    DatasetMeta(
        dataset_path="",
        preprocess_func=ECBenchPreprocessor(),
        tags=['multi-modal', 'vqa']))

