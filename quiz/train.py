# -*- coding: utf-8 -*-

import pandas as pd
import random
from tqdm import tqdm
from omegaconf import OmegaConf

dict_conf = {
    'pretrained_model_name': 'sonoisa/t5-base-japanese',
    'model_dir': 'model',
    'data_dir': 'data',
    'seed': 42,

    'train': True,
    'eval': True,
    
    'max_input_length': 512,
    'max_target_length': 64,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'num_train_epochs': 10,

    'temperature':  1.0,            # 生成にランダム性を入れる温度パラメータ
    'repetition_penalty': 1.5,      # 同じ文の繰り返し（モード崩壊）へのペナルティ
    'num_beams': 10,                # ビームサーチの探索幅
    'diversity_penalty': 1.0,       # 生成結果の多様性を生み出すためのペナルティ
    'num_beam_groups': 10,          # ビームサーチのグループ数
    'num_return_sequences': 10,     # 生成する文の数
}

base_conf = OmegaConf.create(dict_conf)
cli_conf = OmegaConf.from_cli()
conf = OmegaConf.merge(base_conf, cli_conf)
print(conf)


# 事前学習済みモデル
PRETRAINED_MODEL_NAME = conf.pretrained_model_name

# 転移学習済みモデル
MODEL_DIR = conf.model_dir


import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

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

# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(conf.seed)


# GPU利用有無
USE_GPU = torch.cuda.is_available()

# 各種ハイパーパラメータ
args_dict = dict(
    data_dir=conf.data_dir,  # データセットのディレクトリ
    model_name_or_path=PRETRAINED_MODEL_NAME,
    tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    # max_input_length=512,
    # max_target_length=64,
    # train_batch_size=8,
    # eval_batch_size=8,
    # num_train_epochs=4,

    n_gpu=1 if USE_GPU else 0,
    early_stop_callback=False,
    #fp_16=False,
    fp_16=True,
    opt_level='O1',
    max_grad_norm=1.0,
    seed=42,

)

import re
import unicodedata

import neologdn

import json

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


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        try:
            self.hparams = hparams
        except AttributeError:
            self.save_hyperparameters(hparams)

        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path, is_fast=True)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        labels = batch["target_ids"]

        # All labels set to -100 are ignored (masked), 
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.hparams.learning_rate, 
                          eps=self.hparams.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_dataset(self, tokenizer, type_path, args):
        return TsvDataset(
            tokenizer=tokenizer, 
            data_dir=args.data_dir, 
            type_path=type_path, 
            input_max_len=args.max_input_length,
            target_max_len=args.max_target_length)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset = self.get_dataset(tokenizer=self.tokenizer, 
                                             type_path="train.tsv", args=self.hparams)
            self.train_dataset = train_dataset

            val_dataset = self.get_dataset(tokenizer=self.tokenizer, 
                                           type_path="dev.tsv", args=self.hparams)
            self.val_dataset = val_dataset

            self.t_total = (
                (len(train_dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.hparams.train_batch_size, 
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.hparams.eval_batch_size, 
                          num_workers=4)

if True:
    args_dict.update({
        "max_input_length":  conf.max_input_length,  # 入力文の最大トークン数
        "max_target_length": conf.max_target_length,  # 出力文の最大トークン数
        "train_batch_size":  conf.train_batch_size,  # 訓練時のバッチサイズ
        "eval_batch_size":   conf.eval_batch_size,  # テスト時のバッチサイズ
        "num_train_epochs":  conf.num_train_epochs,  # 訓練するエポック数
        })
    args = argparse.Namespace(**args_dict)


if conf.train:
    tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME, is_fast=True)

    # 訓練データセットの読み込み
    train_dataset = TsvDataset(tokenizer, args_dict["data_dir"], "train.tsv", 
                               input_max_len=512, target_max_len=64)

    # チェックポイントほか
    checkpoint_dir = "model/checkpoints"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        checkpoint_dir, 
        monitor="val_loss", mode="min", save_top_k=1
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        devices=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision= 16 if args.fp_16 else 32,
        #amp_level=args.opt_level,
        #amp_backend='apex',
        gradient_clip_val=args.max_grad_norm,
        callbacks=[checkpoint_callback],    
    )

    def find_latest_checkpoints(checkpoint_dir):
        ckpts = sorted(glob.glob(checkpoint_dir+"/*.ckpt"))
        if len(ckpts) == 0:
            return None
        else:
            return ckpts[-1]

    resume_ckpt = find_latest_checkpoints(checkpoint_dir) 

    # 転移学習の実行
    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model, ckpt_path=resume_ckpt)

    # 最終エポックのモデルを保存
    model.tokenizer.save_pretrained(MODEL_DIR)
    model.model.save_pretrained(MODEL_DIR)

    del model


if conf.eval:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True)
    trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        trained_model.cuda()

    import textwrap
    from tqdm import tqdm
    from sklearn import metrics

    test_dataset = TsvDataset(tokenizer, args_dict["data_dir"], "test.tsv", 
                              input_max_len=args.max_input_length, 
                              target_max_len=args.max_target_length)

    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)

    trained_model.eval()

    inputs = []
    outputs = []
    targets = []
    qaids = []

    for batch in tqdm(test_loader):
        input_ids = batch['source_ids']
        input_mask = batch['source_mask']
        qa_ids = batch['qa_id']
        if USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        output = trained_model.generate(input_ids=input_ids, 
            attention_mask=input_mask, 
            max_length=args.max_target_length,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
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

    for output, target, input in zip(outputs, targets, inputs, qaids):
        print("qa_id     : " + output)
        print("generated : " + output)
        print("target    : " + target)
        print("input     : " + input)
        print()

    if True:
        import textwrap
        from tqdm.auto import tqdm
        from sklearn import metrics
        import collections

        test_dataset = TsvDataset(tokenizer, args_dict["data_dir"], "test.tsv", 
                                input_max_len=args.max_input_length, 
                                target_max_len=args.max_target_length)

        test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)

        predictions = collections.OrderedDict()
        outputs = []
        targets = []

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

            outs = trained_model.generate(input_ids=input_ids, 
                attention_mask=input_mask, 
                max_length=args.max_target_length,
                return_dict_in_generate=True,
                output_scores=True)

            dec = decode_to_whitespace_delimited_tokens(outs.sequences)
            target = decode_to_whitespace_delimited_tokens(batch["target_ids"])

            outputs.extend(dec)
            targets.extend(target)

            for qa_id, output in zip(qa_ids, dec):
                predictions[qa_id] = output


        from transformers.data.metrics.squad_metrics import squad_evaluate
        from transformers.data.processors.squad import SquadV2Processor

        processor = SquadV2Processor()
        examples = processor.get_dev_examples("data", filename="normalized-valid-v1.1.json")

        # ## 精度評価
        # 
        # - EM: 文字列が完全一致した割合
        # - F1: トークンが一致した割合のF1値

        results = squad_evaluate(examples, predictions)
        print(f"EM: {results['exact']}\nF1: {results['f1']}")



    del model
