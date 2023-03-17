
# t5のファインチューニング

今までの反省から

- 簡単のため、データの準備段階で入力を整形しておく。プロンプト込みの状態
  - TODO：　キーとプロンプトを指定できれば、任意の形にできるだろう
- データ整形は、改行除去 + NFKC正規化 + neologdn
  - 小文字そろえしない
    - 固有名詞でなければ、製品名でなければ、小文字そろえしてもよいが
  - TODO: かっこの除去はやってよいかも
- 入出力データ形式の変更。
  - TSVはそのまま
  - 旧形式は"{question}\t{answer}\t{context}"
  - 形式は"{QA_ID}\tquestion: {question} context: {context}\t{answer}"
    - QA_IDをキーとしたい

## コード

- download.sh   : データセットのダウンロード
- prepare_jsquad.py : JGLUE/JSQuAD を qa用TSVに加工
  - qid, "question: {question} context: {passage}", answer
- train.py
- predict.py : text2text-generation パイプラインを呼ぶだけ


## 利用

※未実装のものあり

JSQuAD QA

```
'question: {question} context: {context}' -> '{answer}'
```

```bash
download.sh
python prepare_jsquad.py
python train.py
```

JSQuAD AQG

```
'answer: {answer} context: {context}' -> '{question}'
```

```bash
download.sh
python prepare_jsquad_aqg.py
python train.py --max_target_length 128
```

quiz  AQG

```
'answer: {answer} context: {context}' -> '{question}'
```

```bashd
python prepare_quiz_aqg.py
python train.py --max_target_length 128
```

quiz  AQG-HL    
    - https://huggingface.co/p208p2002/t5-squad-qg-hl
    - https://github.com/patil-suraj/question_generation#answer-aware-question-generation

    ハイライト形式。区切れる？
    文を<SEP>で区切る?
    <hl> 42 <hl> is the answer to life, the universe and everything.


```
'c1 c2 ... <hl> a1 ... a|A| <hl> ... c|C|' -> '{question}'
```

解答語句部分をハイライトする。ハイライトされた周辺文脈に基づいて生成するよう誘導する?

```bash
python prepare_quiz_aqg_hl.py
python train.py --max_target_length 128
```

3/17 現状。fp_16=Trueだとnan。fp_16=Falseだとよい。


## 評価

TODO:
- sumevalを使っていたが、huggingface datasets metrics か huggingface evaluateを使うように。
- トークンまたは単語単位での比較が必要で、T5tokenizer、mecab, sudachi(A? B? C?)のそれぞれで区切って比較する。

- `model_name_or_path` でなく `model_dir` を読む
- ExactMatch と sacrebleu[ja]による BLEU

```bash
python train.py --no_train --model_dir model
```


## 推論

※ AQG専用

```
'answer: {answer} context: {context}' -> '{question}'
```

```bash
python infer.py

python infer.py --answer 甲府市 --context 山梨県の県庁所在地は甲府市です。
#山梨県の県庁所在地はどこですか?
python infer.py --answer 甲府市 --context 山梨県の県庁所在地は甲府市です。 --bad_words 山梨 山梨県
#中部地方の県庁所在地はどこですか?
```

## 環境設定

```bash
# python-3.10.8
pip install -qU pip wheel
pip install -qU neologdn pandas numpy scikit-learn tqdm classopt
pip install -qU torch torchtext torchvision torchaudio
pip install -qU transformers pytorch-lightning sentencepiece protobuf==3.20.0
pip install -qU sacrebleu[ja]
```

```
Package                  Version
------------------------ ----------
classopt                 0.2.1
neologdn                 0.5.1
numpy                    1.24.2
pandas                   1.5.3
pip                      23.0.1
protobuf                 3.20.0
pytorch-lightning        2.0.0
scikit-learn             1.2.2
sentencepiece            0.1.97
torch                    2.0.0
torchaudio               2.0.1
torchtext                0.15.1
torchvision              0.15.1
tqdm                     4.65.0
transformers             4.27.1
wheel                    0.40.0
```

## 履歴

- 3/16  全面的に書き直し
- 3/17  omegaconfとargparse併用から omegaconf単体へ。さらにclassopt
  - classoptの場合、notebookなどからの利用が不可?

## TODO

- pytorch_lightningから脱却。素直なループに変更する。
  - 損失を自由に定義できるように
  - これにならう https://github.com/hppRC/bert-classification-tutorial
- いくつか不足のところがありそう
  - sonoisaだけでなくpatil-surajのこれか。これをもとにする。
    - https://github.com/patil-suraj/question_generation
    - https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb
- best (== val_loss最小)のモデルを保存するよう。評価でもそれを用いるよう
  - https://github.com/hppRC/bert-classification-tutorial/blob/main/src/train.py


- ROUGEの実装と
- 普通のループで損失を生に出すところだな。

## ハルシーネーション

- 簡単化のため、ハルシネーションとは生成文に出現する語句のうちで原文または原文章には出現しない語句と定義する。
- 言い換えや表記ゆれの場合、ハルシーネーションとみなす。さもなくば事前修正すべき。
- 背景知識があれば問題ないと判断できる場合でも、ハルシーネーションとみなす。さもなくば事前修正すべき。
- 内容語に限定するのが望ましい。機能語は対象とすべきでなく、対象としても仕方ないだろう。
  - 機能語によって文意が反転することがあるだろうが、ここでは気にしない。
- トークン単位でみる。トーカナイザは mecab か sudachiか。

```python
import spacy

nlp = spacy.load('ja_ginza')

def check_hallucination(candidate, reference):
  doc_c = nlp(candidate)
  doc_r = nlp(reference)

  tok_c = [i.text for i def checkdedefdin doc_c]
  tok_r = [i.text for i in doc_r]

  hallucination = set(tok_c) - set(tok_r)

  # TODO: 内容語のみに限定するか、機能語を除くか、記号を除くか
  return hallucination  

candidate = "今日はわるい天気だ"  # 「わるい」が hallucination
reference = "今日は天気だ"

check_hallucination(candidate="今日はわるい天気だ", reference="今日はわるい天気だ")
# > set()
check_hallucination(candidate="今日は天気だ", reference="今日はわるい天気だ")
# > set()
check_hallucination(candidate="今日はわるい天気だ", reference="今日は天気だ")
# >{'わるい'}
check_hallucination(candidate="今日はわるい天気だ", reference="今日はよい天気だ")
# >{'わるい'}
```
