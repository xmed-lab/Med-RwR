The knowledge base can be downloaded from [Hugging Face](https://huggingface.co/Luxuriant16/Med-RwR/tree/main/knowledge_base).

Option A) huggingface_hub (recommended)
```bash
pip install -U "huggingface_hub>=0.23.0"
python -c 'from huggingface_hub import snapshot_download; \
snapshot_download(repo_id="Luxuriant16/Med-RwR", \
allow_patterns=["knowledge_base/*"], \
local_dir="retrieve/knowledge_base", \
local_dir_use_symlinks=False)'
```

Option B) git + git-lfs
```bash
git lfs install
git clone https://huggingface.co/Luxuriant16/Med-RwR hf-medrwr
cp -R hf-medrwr/knowledge_base/* retrieve/knowledge_base/
rm -rf hf-medrwr
```