# Transformer

I design a poet-gpt using transformers which generates poems. It uses Hugging Face tokenizer.

1. Create an environment: `python -m venv venv`
2. Activate the environment: `source venv/bin/activate` (to deactivate just run `deactivate`)
3. Install requirements: `pip install -r requirements.txt`
4. Run `python -m split` to generate source and target files.
5. Train a model and save it as `models/model.pt`: `python -m train`
