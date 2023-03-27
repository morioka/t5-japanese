# -*- coding: utf-8 -*-

import random
from tqdm import tqdm

from classopt import classopt

import argparse
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from torch.optim import AdamW

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

import unicodedata
import neologdn


# GPU利用有無
USE_GPU = torch.cuda.is_available()

@classopt(default_long=True, default_short=False)
class Args:
    model_name_or_path: str = 'sonoisa/t5-base-japanese'
    tokenizer_name_or_path: str = None

    output_dir: str = 'model'
    data_dir: str = 'data'

    seed: int = 42

    do_train: bool = False
    do_eval: bool = False
    
    max_input_length: int = 512
    max_target_length: int = 64

    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_train_epochs: int = 10

    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1

    n_gpu: int = 1 if USE_GPU else 0
    early_stop_callback: bool = False
    fp16: bool = False
    fp16_opt_level: str = 'O1'
    max_grad_norm: float =1.0

    # 推論時
    # https://huggingface.co/docs/transformers/main_classes/text_generation
    # https://huggingface.co/docs/transformers/generation_strategies
    temperature: float = 1.0           # 生成にランダム性を入れる温度パラメータ  (float, optional, defaults to 1.0) — The value used to modulate the next token probabilities.
    repetition_penalty: float = 1.5    # 同じ文の繰り返し（モード崩壊）へのペナルティ   (float, optional, defaults to 1.0) — The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.
#    num_beams: int = 10                # ビームサーチの探索幅   (int, optional, defaults to 1) — Number of beams for beam search. 1 means no beam search.
#    diversity_penalty: float = 1.0     # 生成結果の多様性を生み出すためのペナルティ    (float, optional, defaults to 0.0) — This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a particular time. Note that diversity_penalty is only effective if group beam search is enabled.
#    num_beam_groups: int = 10          # ビームサーチのグループ数  (int, optional, defaults to 1) — Number of groups to divide num_beams into in order to ensure diversity among different groups of beams. this paper for more details.
#    num_return_sequences: int = 10     # 生成する文の数

    verbose: bool = False               # デバグ用：引数を出力


conf: Args = Args.from_args()
if conf.verbose:
    print(conf)


# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(conf.seed)


def normalize_text(text):
    text = text.strip()
    assert "\t" not in text
    assert "\r" not in text
    assert "\n" not in text
    assert len(text) > 0

    text = neologdn.normalize(unicodedata.normalize('NFKC', text))
    #text = text.lower()
    return text

# ## TSVデータセットクラス
# 
# TSV形式のファイルをデータセットとして読み込みます。  
# 形式は"{QA_ID}\t{question: 質問文 context: コンテキスト}\t{答え}"です。

class TsvDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, input_max_len=512, target_max_len=512):
        self.file_path = os.path.join(data_dir, type_path)
        
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.qa_ids = []

        self._build()
  
    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        qa_id = self.qa_ids[index]

        return {"source_ids": source_ids, "source_mask": source_mask, 
                "target_ids": target_ids, "target_mask": target_mask,
                "qa_id": qa_id}

    def _build(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split("\t")
                assert len(line) == 3
                assert len(line[0]) > 0
                assert len(line[1]) > 0
                assert len(line[2]) > 0

                qa_id = line[0].strip()
                input = line[1].strip()
                target = line[2].strip()

                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input], max_length=self.input_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [target], max_length=self.target_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
                self.qa_ids.append(qa_id)

if conf.do_eval:
    from util import bleu, rouge

    tokenizer = T5Tokenizer.from_pretrained(conf.output_dir, is_fast=True)
    trained_model = T5ForConditionalGeneration.from_pretrained(conf.output_dir)

    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        trained_model.cuda()

    import textwrap
    from tqdm import tqdm
    from sklearn import metrics
    import collections
    
    test_dataset = TsvDataset(tokenizer, conf.data_dir, "test.tsv",     # ここで評価用に加工したtest.tsvを指す
                              input_max_len=conf.max_input_length, 
                              target_max_len=conf.max_target_length)

    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)

    trained_model.eval()

    inputs = []
    outputs = []
    targets = []
    qaids = []

    predictions = collections.OrderedDict()
    examples = collections.OrderedDict()

    def untokenize(ids):
        token_texts = [tokenizer.decode([id], skip_special_tokens=True).strip() for id in ids]
        token_texts = [t for t in token_texts if t != ""]
        return token_texts

    # 注意: JSQuADのF1値計算の都合で、トークンの間に半角空白を入れた文字列に変換する。
    def decode_to_whitespace_delimited_tokens(sequences):
        return [" ".join(untokenize(ids.cpu().tolist())).strip() for ids in sequences]

    for batch in tqdm(test_loader):
        input_ids = batch['source_ids']
        input_mask = batch['source_mask']
        qa_ids = batch['qa_id']
        if USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        output = trained_model.generate(input_ids=input_ids, 
            attention_mask=input_mask, 
            max_length=conf.max_target_length,
            temperature=conf.temperature,
            repetition_penalty=conf.repetition_penalty,
            return_dict_in_generate=True,  # t5_JSQuAD.ipynb にはある
            output_scores=True             # t5_JSQuAD.ipynb にはある
        )

        output_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                                   clean_up_tokenization_spaces=False) 
                    for ids in output]
        target_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                                   clean_up_tokenization_spaces=False) 
                    for ids in batch["target_ids"]]
        input_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                                   clean_up_tokenization_spaces=False) 
                    for ids in input_ids]

        inputs.extend(input_text)
        outputs.extend(output_text)
        targets.extend(target_text)
        qaids.extend(qa_ids)


        ##
        dec = decode_to_whitespace_delimited_tokens(output.sequences)
        target = decode_to_whitespace_delimited_tokens(batch["target_ids"])
        #
        for qa_id, output in zip(qa_ids, dec):
            predictions[qa_id] = output
        for qa_id, tgt in zip(qa_ids, target):
            examples[qa_id] = tgt


    # 一覧出力
    with open(f'{conf.output_dir}/test_qa_output.tsv', 'w') as f:
        f.write('qa_id\tinput\ttarget\toutput\n')
        for output, target, input, qa_id in zip(outputs, targets, inputs, qaids):
            try:
                f.write(f'{qa_id}\t{input}\t{target}\t{output}\n')
            except:
                pass

    # ExactMatch: 文字列が完全一致した割合
    results = {
        "exact": np.array([ o == t for (o, t) in zip(outputs, targets)]).sum() / len(outputs)
    }

    # BLEU: n-gram 一致数を基にした機械翻訳の自動評価指標の一つ
    results['bleu'] = bleu(outputs, targets)

    # ROUGE: n-gram 一致数を基にした機械翻訳の自動評価指標の一つ
    results['rouge'] = rouge(outputs, targets)['rougeAve']

    print(f"EM: {results['exact']}\nBLEU: {results['bleu']}\nROGUE: {results['rouge']}")
    with open(f'{conf.output_dir}/test_qa_summary.txt', 'w') as f:
        f.write(f"EM: {results['exact']}\nBLEU: {results['bleu']}\nROGUE: {results['rouge']}\n")

    if True:
        from transformers.data.metrics.squad_metrics import squad_evaluate
        from transformers.data.processors.squad import SquadV2Processor

#        processor = SquadV2Processor()
#        examples = processor.get_dev_examples("data", filename="normalized-valid-v1.1.json")

        # ## 精度評価
        # 
        # - EM: 文字列が完全一致した割合
        # - F1: トークンが一致した割合のF1値

        results = squad_evaluate(examples, predictions)
        print(f"EM: {results['exact']}\nF1: {results['f1']}")
        with open(f'{conf.output_dir}/test_qa_summary.txt', 'a') as f:
            f.write(f"QA-EM: {results['exact']}\nQA-F1: {results['f1']}")

    # 後始末

