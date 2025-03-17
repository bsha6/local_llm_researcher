### Architecture:
https://www.figma.com/board/5mR3IAIMjT2B8zyFdE2VOJ/ES-%2B-FAISS-Project-for-LLM-Articles?node-id=0-1&t=8A4befJQjABtRJvF-1

### TODO:
generate_image_embeddings()
-> ensure images are only vectorized once

-batch way of generating embeddings? what's the most efficient way to do this
-store in FAISS
-> way of validating this??

-set up LLM
-> connect to FAISS DB (HNSW indexing)
-> connect to SQLite (can use LangChain or some other framework)

-conda env list -> env.yaml
-set up git repo

#### TESTS:
-integration testing (everything works together)

### FUTURE IMPROVEMENTS:
-not getting doi from arxiv (null in DB)... look into this
-> find article with doi and ensure it's being extracted properly

### Setup:
- Make sure to add PYTHONPATH in VSCode settings
"terminal.integrated.env.osx": { "PYTHONPATH": "${workspaceFolder}" }	

### Running Scripts:
-run from root folder using python -m folder.file_name

### Dependencies
-installed ghostscript view homebrew

