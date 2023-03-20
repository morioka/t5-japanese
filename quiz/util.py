

# BLEU
# ROUGE
# hallucination


# はじめての自然言語処理 BRIO による抽象型要約の検証 | オブジェクトの広場 (2022-10-27)
# https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part23.html

import compare_mt.rouge.tokenize as rouge_tokenize
rouge_tokenize.tokenize("I have a pen.", stemmer=False)    

import os
import ipadic
os.environ["MECABRC"] ="/etc/mecabrc"
import MeCab
#mecab = MeCab.Tagger ("-Ochasen")
mecab = MeCab.Tagger(ipadic.MECAB_ARGS + " -Owakati")   # from sacrebleu
mecab.parse("")

def parse_by_mecab(sentence, lemma=False):
  tokens = []
  node = mecab.parseToNode(sentence).next
  while node:
    feature = node.feature.split(',')
    token = feature[-3] # 標準形
    if token == '*' or not lemma:
      token = node.surface
    tokens.append(token)
    node = node.next
  return [token for token in tokens if len(token) > 0]

def tokenize(text, stemmer):
  return parse_by_mecab(text, lemma=False)

rouge_tokenize.tokenize = tokenize

import compare_mt.rouge.tokenize as rouge_tokenize
rouge_tokenize.tokenize("私はペンを持っています。", stemmer=False)
# ['私', 'は', 'ペン', 'を', '持っ', 'て', 'い', 'ます', '。']

from compare_mt.rouge.rouge_scorer import RougeScorer
all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

import spacy
nlp = spacy.load('ja_ginza')

def process(x):
    return [str(sent) for sent in nlp(x.strip()).sents]

#def compute_metrics(preds, labels):
def rouge(preds, labels):
#def compute_metrics(eval_prediction):
    #label_ids = eval_prediction.label_ids
    #pred_ids = eval_prediction.predictions
    #labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    labels = [process(label) for label in labels]
    preds = [process(pred) for pred in preds]
    sample_rouge1 = 0
    sample_rouge2 = 0
    sample_rougeLsum = 0
    cnt=0
    for pred, label in zip(preds, labels):
      score = all_scorer.score("\n".join(label), "\n".join(pred))
      sample_rouge1 += score["rouge1"].fmeasure
      sample_rouge2 += score["rouge2"].fmeasure
      sample_rougeLsum += score["rougeLsum"].fmeasure
      cnt += 1
    sample_rouge1 = sample_rouge1 / cnt
    sample_rouge2 = sample_rouge2 / cnt
    sample_rougeLsum = sample_rougeLsum / cnt
    sample_rougeAve = (sample_rouge1 + sample_rouge2 + sample_rougeLsum) / 3.0
    return {"rouge1": sample_rouge1, "rouge2": sample_rouge2, "rougeLsum": sample_rougeLsum, "rougeAve":  sample_rougeAve}

#compute_metrics(["今日は天気です"]
#print(rouge(["今日は天気です"]  ,["今日はよい天気です"]))


# はじめての自然言語処理 Hugging Face Transformers で T5 を使ってみる | オブジェクトの広場 (2021-04-22)
# https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part14.html
from sacrebleu import corpus_bleu

def bleu(predictions, references):
    references = [references]
    bleu_score = corpus_bleu(predictions, references,      
                                    smooth_method="exp",
                                    smooth_value=0.0,
                                    force=False,
                                    lowercase=False,
                                    tokenize="ja-mecab",
                                    use_effective_order=False)
    return bleu_score.score

#
# ハルシネーション
# 簡単のため、生成文側に出現する語句のうち、参照文に出現しないものとする。
# あるべきは内容語に注目すること。機能語は考慮しなくてよいだろう。
# 言い換えや表記ゆれを認識しない。基本的または常識的な知識で解決できる場合も認識しない。

HALLCINATION_STOP_WORDS = ['でしょう', 'どんな', 'どちら', 'さて']

def check_hallucination(candidate, reference, stop_words=None):
  if stop_words is None:  # [] が指定されると素通し
    stop_words = HALLCINATION_STOP_WORDS

  if True:  # mecab を利用
    tok_c = tokenize(candidate, stemmer=False)
    tok_r = tokenize(reference, stemmer=False)
  else:
    doc_c = nlp(candidate)
    doc_r = nlp(reference)

    tok_c = [i.text for i in doc_c]
    tok_r = [i.text for i in doc_r]

  hallucination = set(tok_c) - set(tok_r)

  # stop_wordsに含まれる語はハルシネーションとみなさない
  # TODO: 内容語のみに限定するか、機能語を除くか、記号を除くか
  if type(stop_words) is list:
    stop_words = set(stop_words)
  assert type(stop_words) is set
  hallucination = hallucination - stop_words
  
  return hallucination  

# 検査：解答が質問文にあるか
# あるとNG
def check_ansewr_in_question(answer, question):
  return answer in question

# 検査：質問文＋解答とコンテキストとの類似度
def check_similarity_question_context(answer, question, context):
  return nlp(f"{question}答えは{answer}").similarity(nlp(context))

# 検査；質問文＋解答はコンテキストを含意するか
# NLIモデルを使いたい。重いが。
# https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part18.html
# https://axross-recipe.com/recipes/320
# https://aclanthology.org/2020.acl-main.272/
def check_nli(answer, question, context):
  return False  # NOT_IMPLEMENTED


# ユーティリティ: LEAD-3 要約のベースライン
def lead3(text, lines=3):
  return "".join([sent.text for sent in nlp(text).sents][:lines])


###

def test_check_hallucination():
  candidate = "今日はわるい天気だ"  # 「わるい」が hallucination
  reference = "今日は天気だ"

  assert check_hallucination(candidate="今日はわるい天気だ", reference="今日はわるい天気だ") == set()
  assert check_hallucination(candidate="今日は天気だ", reference="今日はわるい天気だ") == set()
  assert check_hallucination(candidate="今日はわるい天気だ", reference="今日は天気だ") == set(['わるい'])
  assert check_hallucination(candidate="今日はわるい天気だ", reference="今日はよい天気だ") == set(['わるい'])
  assert check_hallucination(candidate="今日はわるい天気だ", reference="今日はよい天気だ", stop_words=['わるい']) == set()

def test_bleu():
  candidate = "今日はわるい天気だ"  # 「わるい」が hallucination
  reference = "今日は天気だ"

  # bleu([candidate], [reference])) == 30.213753973567677
  assert bleu([candidate], [reference]) > 30.213
  assert bleu([candidate], [reference]) < 30.214

def test_rogue():
  candidate = "今日はわるい天気だ"  # 「わるい」が hallucination
  reference = "今日は天気だ"

  # rouge([candidate], [reference]) == {'rouge1': 0.888888888888889, 
  #                                     'rouge2': 0.5714285714285715,
  #                                     'rougeLsum': 0.888888888888889,
  #                                     'rougeAve': 0.7830687830687831}

  # bleu([candidate], [reference])) == 30.213753973567677
  assert rouge([candidate], [reference])['rougeAve'] > 0.7830
  assert rouge([candidate], [reference])['rougeAve'] < 0.7831


def test_lead3():
  text = '今日はいい天気。明日もいい天気。昨日は悪い天気。来年はわからない。'
  
  assert lead3(text) == '今日はいい天気。明日もいい天気。昨日は悪い天気。'
  assert lead3(text, lines=5) == text
  assert lead3(text, lines=2) == '今日はいい天気。明日もいい天気。'

if __name__ == '__main__':
  test_check_hallucination()
  test_bleu()
  test_rogue()
  test_lead3()
  print('DONE')
