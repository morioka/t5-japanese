
# t5のファインチューニング

- 簡単のため、データの準備段階で入力を整形しておく。プロンプト込みの状態
  - TODO：　キーとプロンプトを指定できれば、任意の形にできるだろう
- データ整形は、改行除去 + NFKC正規化 + neologdn
  - 小文字そろえしない
- 入出力データ形式の変更。
  - TSVはそのまま
  - 旧形式は"{question}\t{answer}\t{context}"
  - 形式は"{QA_ID}\tquestion: {question} context: {context}\t{answer}"
    - QA_IDをキーとしたい


## コード

- download.sh   : データセットのダウンロード
- prepare_data_jsquad_qa.py : JGLUE/JSQuAD を qa用TSVに加工
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
python prepare_data_jsquad_qa.py
python train.py
```

JSQuAD AQG

```
'answer: {answer} context: {context}' -> '{question}'
```

```bash
download.sh
python prepare_data_jsquad_aqg.py
python train.py max_target_length=128
```

quiz  AQG

```
'answer: {answer} context: {context}' -> '{question}'
```

```bashd
python prepare_quiz_aqg.py
python train.py max_target_length=128
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
python train.py max_target_length=128
```

## 評価

TODO:
- sumevalを使っていたが、huggingface datasets metrics か huggingface evaluateを使うように。
- トークンまたは単語単位での比較が必要で、T5tokenizer、mecab, sudachi(A? B? C?)のそれぞれで区切って比較する。

---
sonoisaだけでなくpatil-surajのこれか。これをもとにする。
https://github.com/patil-suraj/question_generation
https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb

そのあと、損失に手を入れられるよう、pytorch lightning依存をなくす。


## 推論

```bash
python predict.py
```

```bash
python predict.py --help
```

```bash
# python-3.10.8
pip install -qU pip wheel
pip install -qU neologdn pandas numpy scikit-learn tqdm classopt
pip install -qU torch torchtext torchvision torchaudio
pip install -qU transformers pytorch-lightning sentencepiece protobuf==3.20.0
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