#!/bin/bash

# JGLUE/JSQuAD-1.1.0
wget https://github.com/yahoojapan/JGLUE/archive/refs/tags/v1.1.0.zip
unzip v1.1.0.zip
rm v1.1.0.zip

#  run_clm.py : transformers 4.27.0.dev0 (running on 4.27.1)
wget https://raw.githubusercontent.com/huggingface/transformers/b19d64d852804b6bf36763f8429352cf7b5ce0cb/examples/pytorch/language-modeling/run_clm.py run_clm.py

