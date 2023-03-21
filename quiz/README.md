
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
python train.py --do_train --do_eval \
  --model_name_or_path sonoisa/t5-base-japanese 
```

JSQuAD AQG

```
'answer: {answer} context: {context}' -> '{question}'
```

```bash
download.sh
python prepare_jsquad_aqg.py
python train.py --do_train --do_eval \
  --model_name_or_path sonoisa/t5-base-japanese \
  --output_dir model_jsquad_aqg \
  --data_dir data_jsquad_aqg \
  --max_target_length 128
```


JSQuAD AQG-HL

```
'c1 c2 ... <hl> a1 ... a|A| <hl> ... c|C|' -> '{question}'
```

```bash
download.sh
python prepare_jsquad_aqg.py  # 中の with_highlight=True
python train.py --do_train --do_eval \
  --model_name_or_path sonoisa/t5-base-japanese \
  --output_dir model_jsquad_aqg_hl \
  --data_dir data_jsquad_aqg_hl \
  --max_target_length 128
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
    文を<SEP>で区切る? 文ごとにattention_maskを変える?
    <hl> 42 <hl> is the answer to life, the universe and everything.


```
'c1 c2 ... <hl> a1 ... a|A| <hl> ... c|C|' -> '{question}'
```

解答語句部分をハイライトする。ハイライトされた周辺文脈に基づいて生成するよう誘導する?

```bash
python prepare_quiz_aqg_hl.py
python train.py --max_target_length 128
```

3/17 現状。fp16=Trueだとnan。fp16=Falseだとよい。


## 評価

TODO:
- sumevalを使っていたが、huggingface datasets metrics か huggingface evaluateを使うように。
- トークンまたは単語単位での比較が必要で、T5tokenizer、mecab, sudachi(A? B? C?)のそれぞれで区切って比較する。

- `model_name_or_path` でなく `output_dir` を読む
- ExactMatch と sacrebleu[ja]による BLEU

```bash
python train.py --do_eval --output_dir model
```

### 評価指標

- EM (ExactMatch):  文字列ベース
- BLUE:   mjpost/sacrebleu + mecab-ipadic
- ROUGE:  neulab/compare-mt + mecab-ipadic 

## 推論

※ AQG専用

```
'answer: {answer} context: {context}' -> '{question}'
```

```bash
python generate.py

python generate.py --answer 甲府市 --context 山梨県の県庁所在地は甲府市です。
#山梨県の県庁所在地はどこですか?
python generate.py --answer 甲府市 --context 山梨県の県庁所在地は甲府市です。 --bad_words 山梨 山梨県
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
pip install -qU compare_mt
pip install -qU mecab-python3 fugashi ipadic
pip install -qU spacy ja_ginza
pip install -qU datasets evaluate
```

```
Package                  Version
------------------------ ----------
classopt                 0.2.1
compare-mt               0.2.10
fugashi                  1.2.1
ginza                    5.1.2
ipadic                   1.0.0
ja-ginza                 5.1.2
mecab-python3            1.0.5
neologdn                 0.5.1
numpy                    1.24.2
pandas                   1.5.3
pip                      23.0.1
protobuf                 3.20.0
pytorch-lightning        2.0.0
scikit-learn             1.2.2
sentencepiece            0.1.97
spacy                    3.4.4
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
- 3/20  bleu, rogue 書き直し

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


- 済)ROUGEの実装と
- 普通のループで損失を生に出すところだな。

## ハルシーネーション

- 簡単化のため、ハルシネーションとは生成文に出現する語句のうちで原文または原文章には出現しない語句と定義する。
- 言い換えや表記ゆれの場合、ハルシーネーションとみなす。さもなくば事前修正すべき。
- 背景知識があれば問題ないと判断できる場合でも、ハルシーネーションとみなす。さもなくば事前修正すべき。
- 内容語に限定するのが望ましい。機能語は対象とすべきでなく、対象としても仕方ないだろう。
  - 機能語によって文意が反転することがあるだろうが、ここでは気にしない。
- トークン単位でみる。トーカナイザは mecab または GiNZA(sudachi)。

```python
import spacy

nlp = spacy.load('ja_ginza')

def check_hallucination(candidate, reference, stop_words=[]):
  doc_c = nlp(candidate)
  doc_r = nlp(reference)

  tok_c = [i.text for i def checkdedefdin doc_c]
  tok_r = [i.text for i in doc_r]

  hallucination = set(tok_c) - set(tok_r)

  # stop_wordsに含まれる語はハルシネーションとみなさない
  # TODO: 内容語のみに限定するか、機能語を除くか、記号を除くか
  if type(stop_words) is list:
    stop_words = set(stop_words)
  assert type(stop_words) is set
  hallucination = hallucination - stop_words
  
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

# TODO 3/20

- pytorch_lightningからの脱却。少なくとも損失を自由に定義。
- 済：データセットの整備  '[SEP]' の扱いを '[sep]' と同様に。
- GPT-2ファインチューニング。じつはrun_clm.py単独で動かせるのでは?
  - GPT-2用のデータセット整備
  - GPT-2モデルせめて1Bできれば6B。そこにLoRAとRF
    - t5-baseよりは大きなパラメタで何とかしたい。mt5でもよいが。
  - [GPT-2をファインチューニングしてニュース記事のタイトルを条件付きで生成してみた。 - Qiita](https://qiita.com/m__k/items/36875fedf8ad1842b729)


## gpt

2023-03-21


- [GPT-2をファインチューニングしてニュース記事のタイトルを条件付きで生成してみた。 - Qiita](https://qiita.com/m__k/items/36875fedf8ad1842b729)

transformersをソースからインストールしなくとも、run_clm.pyだけ持ってきても動いた。

```bash
download.sh
#python prepare_data_jsquad_aqg_hl.py      # t5向けhl形式
python prepare_data_jsquad_aqg_hl_gpt.py  # t5向けからgpt向けに変換
python run_clm.py \
    --model_name_or_path=rinna/japanese-gpt2-medium \
    --train_file=data_jsquad_aqg_hl_gpt/train.txt \
    --validation_file=data_jsquad_aqg_hl_gpt/dev.txt  \
    --do_train \
    --do_eval \
    --num_train_epochs=10 \
    --save_steps=10000 \
    --save_total_limit=3 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --output_dir=output_gpt/ \
    --use_fast_tokenizer=False
```

訓練サイズ=70040, バッチサイズ=1 で 12h以上 /epochと言ってくる。
動かすのが精いっぱい。いけそうと判断したら、もっと大きな環境で動かさないと、まともに学習できない。

- https://twitter.com/morioka/status/1637951549015232512
  - <blockquote class="twitter-tweet"><p lang="en" dir="ltr">I&#39;ve gotten some requests about the &quot;building language models&quot; project from last year&#39;s Stanford Large Language Models class, so we&#39;re releasing it: <a href="https://t.co/UVT0Hdm0mr">https://t.co/UVT0Hdm0mr</a><br><br>The task is to finetune LMs to give them new capabilities/properties, similarly to Toolformer and Alpaca. <a href="https://t.co/RJhuKZLayI">pic.twitter.com/RJhuKZLayI</a></p>&mdash; Sang Michael Xie (@sangmichaelxie) <a href="https://twitter.com/sangmichaelxie/status/1637834223699783680?ref_src=twsrc%5Etfw">March 20, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
- [sangmichaelxie/cs324_p2: Project 2 (Building Large Language Models) for Stanford CS324: Understanding and Developing Large Language Models (Winter 2022](https://github.com/sangmichaelxie/cs324_p2)

---

大きめのモデルで見たほうがよいのかな。
Alpaca-LoRA (6B)で日本語でファインチューニングする記事が出始めた。

"""
筆者の使用GPUはRTX3080一台です。バッチサイズを調整すればGPUメモリが12GB以上であれば単一のGPUでFineTuningが可能だと思います。
""" という記述もある。

- https://twitter.com/morioka/status/1637933189028261890
  - <blockquote class="twitter-tweet"><p lang="ja" dir="ltr">記事を投稿しました！ Alpaca-loraを日本語タスクでファインチューニングする [Python] on <a href="https://twitter.com/hashtag/Qiita?src=hash&amp;ref_src=twsrc%5Etfw">#Qiita</a> <a href="https://t.co/aQS68kt1kQ">https://t.co/aQS68kt1kQ</a></p>&mdash; toshi_456 (@tech_nichijo) <a href="https://twitter.com/tech_nichijo/status/1637420497561583616?ref_src=twsrc%5Etfw">March 19, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
  - [Alpaca-loraを日本語タスクでファインチューニングする - Qiita](https://qiita.com/toshi_456/items/280efc31950ddb083286)
- https://twitter.com/morioka/status/1637831647700848641
  - <blockquote class="twitter-tweet"><p lang="ja" dir="ltr">「手元で動く軽量の大規模言語モデルを日本語でファインチューニングしてみました(Alpaca-LoRA)」という記事を書いてみました。手元で動かせるのはすごく今後の可能性を感じますね。<a href="https://t.co/wOpTX3kJHz">https://t.co/wOpTX3kJHz</a></p>&mdash; Masa Kazama (@masa_kazama) <a href="https://twitter.com/masa_kazama/status/1637703014793510913?ref_src=twsrc%5Etfw">March 20, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
  - [手元で動く軽量の大規模言語モデルを日本語でファインチューニングしてみました(Alpaca-LoRA)｜masa_kazama｜note(https://note.com/masa_kazama/n/nabaa6dfec741)
- https://twitter.com/kun1em0n/status/1637973352777404417
  - <blockquote class="twitter-tweet"><p lang="ja" dir="ltr">日本語で学習させたJapanese-Alpaca-LoRAの13B学習途中(128トークンで学習)。同じ回答を繰り返すので命令で工夫(1、2枚目)。512トークンで学習させた7Bは流暢だけど回答の中身は適当なものもあり(3枚目は適当、4枚目はぼちぼち)。やはり学習時間かかるけど13Bを512トークンで学習させるのが一番か🤔 <a href="https://t.co/RrGnv2CLPB">https://t.co/RrGnv2CLPB</a> <a href="https://t.co/mDl2iU3GZP">pic.twitter.com/mDl2iU3GZP</a></p>&mdash; クニえもん.inc🤗 (@kun1em0n) <a href="https://twitter.com/kun1em0n/status/1637973352777404417?ref_src=twsrc%5Etfw">March 21, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>


