"""Write wikitext-2-raw-v1 test split as one raw text file (the same content as the classic
wiki.test.raw used by llama-perplexity), from the HF parquet (Salesforce/wikitext).

Run in WSL: /root/venv-ref/bin/python bench/quality/make_wikitext.py \
    models/baselines/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet bench/quality/wikitext-2-raw-test.txt
Rows already end in "\n"; they are concatenated unchanged.
"""
import hashlib
import sys

import pyarrow.parquet as pq

rows = pq.read_table(sys.argv[1]).column("text").to_pylist()
text = "".join(rows)
with open(sys.argv[2], "w", encoding="utf-8", newline="") as f:
    f.write(text)
print(len(rows), "rows", len(text), "chars", "sha256", hashlib.sha256(text.encode()).hexdigest())
