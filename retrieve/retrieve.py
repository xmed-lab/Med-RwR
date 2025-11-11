from flask import Flask, request, jsonify
import faiss
import numpy as np
from FlagEmbedding import FlagModel
import os
import time
import sys
import argparse

def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        corpus = file.readlines()
        corpus = [line.strip("\n") for line in corpus]
    return corpus

app = Flask(__name__)

@app.route("/queries", methods=["POST"])
def query():
    data = request.json

    queries = data["queries"]

    k = data.get("k", 3)

    # s = time.time()
    query_embeddings = model.encode_queries(queries)

    all_answers = []
    D, I = index.search(query_embeddings, k=k)  # 假设返回前3个结果
    for idx in I:
        answers_for_query = [corpus[i] for i in idx[:k]] # 找出该query对应的k个答案
        all_answers.append(answers_for_query)  # 将该query的答案列表存储

    return jsonify({"queries": queries, "answers": all_answers})


if __name__ == "__main__":
    # data_type = sys.argv[1]
    port = 5000

    model = FlagModel(
        "BAAI/bge-m3",
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=False,
    )

    print("Loading model completed.")

    file_path = "retrieve/knowledge_base/knowledge_base.tsv"
    corpus = load_corpus(file_path)

    print(f"Loaing corpus with {len(corpus)} samples completed.")

    index_path = "retrieve/knowledge_base/knowledge_base.bin"
    index = faiss.read_index(index_path)

    print("Index has been built.")

    app.run(host="0.0.0.0", port=port, debug=False)  # 在本地监听端口5003
    print("Start to query.")
