#!/usr/bin/env python3
import os
import sys
from typing import List, Dict, Any

import torch

from swift.llm.infer.infer_engine.pt_engine_retrieve_infer import PtEngine
from swift.llm.infer.protocol import RequestConfig
from swift.llm.template.template_inputs import InferRequest

SYSTEM_PROMPT = """You are an experienced expert in medicine. You are given a question, an image and a list of choices. You are required to select the correct answer from the choices.
First observe the image, think about the question and each choice within <think> </think> tags. During thinking, if needed, retrieve medical knowledge using <query> </query> tags. Only one query is allowed. An external agent will retrieve information and return it within <retrieve> </retrieve> tags. 
You can use the retrieved information to continue thinking and further query if more information is needed. When you can reach a conclusion, output your answer within <answer> </answer> tags.
The output should be in the following format:
1. If you need more information, output <think> ... </think>\n<query> ... </query>\n<retrieve> ... </retrieve>\n (Multiple think-query-retrieve cycles may occur)
2. If you can directly reach a conclusion without query, output <think> ... </think>\n<answer> ... </answer>"""

def build_messages(user_text, image):
    if image:
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }, 
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def run_demo(model=None, message=None, image=None, max_new_tokens=512, temperature=0.6, top_p=None, top_k=None, repetition_penalty=None, attn_impl="flash_attn", device_map="cuda"):
    # Built-in defaults so caller can just run run_demo() without args

    engine = PtEngine(
        model_id_or_path=model,
        attn_impl=attn_impl,
        device_map=device_map,
        max_batch_size=1,
    )

    messages = build_messages(message, image)
    infer_request = InferRequest(messages=messages)

    request_cfg = RequestConfig(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    outputs = engine.infer([infer_request], request_cfg, template=engine.default_template, use_tqdm=False)
    first = outputs[0]
    text = first.choices[0].message.content
    print(text)


if __name__ == "__main__":
    
    model_path = "Luxuriant16/MedRwR"
    message = "" ## Put question here
    image = "" ## Put image path here

    run_demo(model=model_path, image=image, message=message)


