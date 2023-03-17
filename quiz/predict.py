# -*- coding: utf-8 -*-

import random
from omegaconf import OmegaConf

import argparse
import random

import numpy as np
from transformers import pipeline

import unicodedata
import neologdn


dict_conf = {
    'pretrained_model_name': 'sonoisa/t5-base-japanese-question-generation',
#    'data_dir': 'data',
    'seed': 42,
    
    'max_input_length': 512,
    'max_target_length': 64,

    'temperature':  1.0,            # 生成にランダム性を入れる温度パラメータ
    'repetition_penalty': 1.5,      # 同じ文の繰り返し（モード崩壊）へのペナルティ
    'num_beams': 10,                # ビームサーチの探索幅
    'diversity_penalty': 1.0,       # 生成結果の多様性を生み出すためのペナルティ
    'num_beam_groups': 10,          # ビームサーチのグループ数
    'num_return_sequences': 10,     # 生成する文の数
}

def normalize_text(text):
    text = text.strip()
    assert "\t" not in text
    assert "\r" not in text
    assert "\n" not in text
    assert len(text) > 0

    text = neologdn.normalize(unicodedata.normalize('NFKC', text))
    #text = text.lower()
    return text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
#    torch.manual_seed(seed)
#    if torch.cuda.is_available():
#        torch.cuda.manual_seed_all(seed)


def main():
    base_conf = OmegaConf.create(dict_conf)
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(base_conf, cli_conf)
    print(conf)

    # 事前学習済みモデル
    PRETRAINED_MODEL_NAME = conf.pretrained_model_name

    # 各種ハイパーパラメータ
    args_dict = dict(
        model_name_or_path=PRETRAINED_MODEL_NAME,
        tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

        seed=conf.seed,

        max_input_length=conf.max_input_length,
        max_target_length=conf.max_target_length,

        temperature=conf.temperature,
        repetition_penalty=conf.repetition_penalty,
        num_beams=conf.num_beams,
        diversity_penalty=conf.diversity_penalty,
        num_beam_groups=conf.num_beam_groups,
        num_return_sequences=conf.num_return_sequences,
    )
    args = argparse.Namespace(**args_dict)

    set_seed(args.seed)

    text2text_question_generaton_prompt = "answer: {answer} context: {context}"

    answer = "富士山"
    context = "富士山は静岡県と山梨県にまたがっている山です。"

    answer = normalize_text(answer.replace("\n", " "))
    context = normalize_text(context.replace("\n", " "))

    text2text_generator = pipeline("text2text-generation", model=PRETRAINED_MODEL_NAME)

    # 単純な禁句処理
    bad_words = [answer]    # 類義語や言い換え、一部の表記抜けや過剰な表記を考慮すべき。外部注入させる?
    bad_words_ids = [text2text_generator.tokenizer(bad_word, add_special_tokens=False).input_ids[1:] for bad_word in bad_words]
    if len(bad_words_ids) == 0:
        bad_words_ids = None

    generated = text2text_generator(
                    text2text_question_generaton_prompt.format(answer=answer, context=context),
                    # 以下、 generate() への入力
                    max_length=args.max_target_length ,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                    diversity_penalty=args.diversity_penalty,
                    num_beam_groups=args.num_beam_groups,
                    num_return_sequences=args.num_return_sequences,
                    repetition_penalty=args.repetition_penalty,
                    bad_words_ids=bad_words_ids
                )[0]['generated_text']

    print(generated)

if __name__ == "__main__":
    main()
