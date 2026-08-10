"""Operational scripts: fetch the corpus, annotate datasets, publish subsets.

These are the parts that touch the outside world -- public archive APIs and the
Hugging Face Hub -- and they are kept out of the ``dendro`` package on purpose.
``import dendro`` should stay cheap and dependency-light; anything that needs
``datasets``, ``huggingface_hub``, or a network round trip lives here.
"""
