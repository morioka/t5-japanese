

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
rouge(["今日は天気です"]                
                 ,["今日はよい天気です"])


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
