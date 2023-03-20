# ## JGLUEコーパスのダウンロード

# https://github.com/sonoisa/t5-japanese/blob/main/t5_JSQuAD.ipynb
# を修正
# 修正点:
#   - 前処理を NKFC正規化+neologdnに。小文字化しない。

#!wget https://github.com/yahoojapan/JGLUE/archive/refs/tags/v1.1.0.zip
#!unzip v1.1.0.zip
#!rm v1.1.0.zip

import os
import re
import unicodedata

import neologdn

import json

import spacy
import numpy as np

with_hl = True

#
nlp = spacy.load('ja_ginza')
                        
def normalize_text(text):
    text = text.strip()
    assert "\t" not in text
    assert "\r" not in text
    assert "\n" not in text
    assert len(text) > 0

    text = neologdn.normalize(unicodedata.normalize('NFKC', text))
    #text = text.lower()
    return text

def make_squad_data(json_data):
    data = []
    for datum in json_data["data"]:
        for paragraph in datum["paragraphs"]:
            context = paragraph["context"]
            context = normalize_text(context).replace("[sep]", "<|n|>")
            for qa in paragraph["qas"]:
                qa_id = qa["id"]

                question = qa["question"]
                question = normalize_text(question)

                answer_text = qa["answers"][0]["text"]
                answer_text = normalize_text(answer_text)
                
                # question-answering または machine-reading-comprehension
#                input = f"question: {question} context: {context}"
#                target = f"{answer_text}"

                # answer-aware question-generation
                input = f"answer: {answer_text} context: {context}"
                target = f"{question}"

                if with_hl:    # answer-aware question-generation with highlight
                    if answer_text not in context:
                        continue

                    #  簡単のために、context中の複数の文を結合・合成してquestionが生成されることを仮定しない。
                    input = f"context: {context.replace(answer_text, f'<hl>answer_text<hl>')}"
                    target = f"{question}"

                    if True:  # answer_textが出現するcontextの文の中でquestionに最も近いものを選ぶ。
                        # spacyの場合はword_embeddingの平均なので、いちいち文ごとのembeddingを取らなくてもよいか。しかしSentはembeddingを持たない
                        question_doc = nlp(question)
                        sents = [str(sent) for sent in nlp(context.strip()).sents]
                        sents_simil = [question_doc.similarity(nlp(sent)) for sent in sents]
                        sents_simil = np.array([simil if answer_text in sent else 0.0  for (sent, simil) in zip(sents, sents_simil)])
                        sent_idx = np.argmax(sents_simil)
                        sents[sent_idx] = sents[sent_idx].replace(answer_text, f'<hl>{answer_text}<hl>')
                        context = "".join(sents)

                        input = f"context: {context}"
                        target = f"{question}"

                # ?? 文を区切るようattention maskを設定してやるべきだろうか。
                data.append((qa_id, input, target))
    return data

#

import json
from tqdm import tqdm
from transformers import T5Tokenizer

# 注意: JSQuADのF1値計算の都合で、トークンの間に半角空白を入れた文字列に変換する。
def decode_to_whitespace_delimited_tokens(tokenizer, ids):
    tokens = [tokenizer.decode([id], skip_special_tokens=True).strip() for id in ids]
    tokens = [token for token in tokens if token != ""]
    return " ".join(tokens).strip()


def normalize_squad_test_data(json_data, model="sonoisa/t5-base-japanese"):
    tokenizer = T5Tokenizer.from_pretrained(model)

    for datum in tqdm(json_data["data"]):
        for paragraph in datum["paragraphs"]:
            context = paragraph["context"]
            context = normalize_text(context).replace("[sep]", "<|n|>")
            paragraph["context"] = context

            for qa in paragraph["qas"]:
                question = qa["question"]
                question = normalize_text(question)

                question_ids = tokenizer.encode(question)
                question = decode_to_whitespace_delimited_tokens(tokenizer, question_ids)

                qa["question"] = question

                for answer in qa["answers"]:
                    answer_text = answer["text"]
                    answer_text = normalize_text(answer_text)

                    answer_ids = tokenizer.encode(answer_text)
                    answer_text = decode_to_whitespace_delimited_tokens(tokenizer, answer_ids)

                    answer["text"] = answer_text


# ## データ分割
# 
# データセットを95% : 5%の比率でtrain/devに分割します。
# 
# * trainデータ: 学習に利用するデータ
# * devデータ: 学習中の精度評価等に利用するデータ
# * testデータ: 学習結果のモデルの精度評価に利用するデータ

import random
from tqdm import tqdm

def assert_field(field):
    assert len(field) > 0
    assert "\t" not in field
    assert "\n" not in field
    assert "\r" not in field

def to_line(data):
    qa_id, input, target = data
    qa_id = qa_id.strip()
    input = input.strip()
    target = target.strip()

    assert_field(qa_id)
    assert_field(input)
    assert_field(target)

    return f"{qa_id}\t{input}\t{target}\n"



def main():
    
    DATA_DIR="data"
    DATA_DIR="data_jsquad_aqg"
    if with_hl:
        DATA_DIR="data_jsquad_aqg_hl"

    os.makedirs(DATA_DIR, exist_ok=True)

    # squad json to tsv 
    with open("JGLUE-1.1.0/datasets/jsquad-v1.1/train-v1.1.json", "r", encoding="utf-8") as f_in:
        json_data = json.load(f_in)
        train_data = make_squad_data(json_data)

    with open("JGLUE-1.1.0/datasets/jsquad-v1.1/valid-v1.1.json", "r", encoding="utf-8") as f_in:
        json_data = json.load(f_in)
        test_data = make_squad_data(json_data)

    # 注意: JSQuADのF1値計算の都合で、トークンの間に半角空白を入れた文字列に変換する。
    PRETRAINED_MODEL_NAME="sonoisa/t5-base-japanese"
    
    with open("JGLUE-1.1.0/datasets/jsquad-v1.1/valid-v1.1.json", "r", encoding="utf-8") as f_in, \
        open(f"{DATA_DIR}/normalized-valid-v1.1.json", "w", encoding="utf-8") as f_test:

        json_data = json.load(f_in)
        normalize_squad_test_data(json_data, model=PRETRAINED_MODEL_NAME)
        json.dump(json_data, f_test, ensure_ascii=False)

    # train/valid/test split
    random.seed(1234)
    random.shuffle(train_data)

    data_size = len(train_data)
    train_ratio = 0.95

    with open(f"{DATA_DIR}/train.tsv", "w", encoding="utf-8") as f_train, \
        open(f"{DATA_DIR}/dev.tsv", "w", encoding="utf-8") as f_dev, \
        open(f"{DATA_DIR}/test.tsv", "w", encoding="utf-8") as f_test:
        
        for i, data in tqdm(enumerate(train_data)):
            line = to_line(data)
            if i < train_ratio * data_size:
                f_train.write(line)
            else:
                f_dev.write(line)
        
        for i, data in tqdm(enumerate(test_data)):
            line = to_line(data)
            f_test.write(line)


if __name__ == "__main__":
    main()
