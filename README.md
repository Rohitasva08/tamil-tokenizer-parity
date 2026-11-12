# Tokenizer parity for English–Tamil

Companion code for the paper "Natural Language Processing in the Era of Large
Language Models: Architectures, Post-Training and Challenges, with a
Tamil-Centred Analysis of Low-Resource and Indian-Language NLP".

Measures how many tokens different tokenizers need for Tamil versus English,
using the FLORES-200 devtest split (1,012 parallel sentences).

## Files
- tokenizer_parity.py: the measurement script
- results.csv: full results
- results_table_rows.txt: the rows used in Table III of the paper
- results_meta.json: library versions used

## How to reproduce
1. Download FLORES-200 from the official FLORES repository (it is not included here).
2. pip install tiktoken transformers sentencepiece protobuf numpy
3. python tokenizer_parity.py --en eng_Latn.devtest --ta tam_Taml.devtest --out results

Llama 3 and Gemma 2 need a Hugging Face login and were not evaluated.
