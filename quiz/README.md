
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

```bash
download.sh
python prepare_data_jsquad_qa.py
python train.py
```

JSQuAD AQG

```bash
download.sh
python prepare_data_jsquad_aqg.py
python train.py max_target_length=128
```

quiz  AQG

```bashd
python prepare_quiz_aqg.py
python train.py max_target_length=128
```

quiz  AQG-HL    
    - https://huggingface.co/p208p2002/t5-squad-qg-hl
    - https://github.com/patil-suraj/question_generation#answer-aware-question-generation

    ハイライト形式。区切れる？
    <hl> 42 <hl> is the answer to life, the universe and everything.

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

--

https://github.com/huggingface/transformers/issues/17504
bad_word_idsをつけるか

## 推論

```bash
python predict.py
```


