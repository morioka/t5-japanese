# -*- coding: utf-8 -*-

import random
import numpy as np

from transformers import pipeline

import unicodedata
import neologdn
import mojimoji

from classopt import classopt

@classopt(default_long=True, default_short=False)
class Args:
    model_name_or_path: str = 'sonoisa/t5-base-japanese-question-generation'
    tokenizer_name_or_path: str = None
    seed: int = 42

    max_input_length: int = 512
    max_target_length: int = 64

    temperature: float = 1.0           # 生成にランダム性を入れる温度パラメータ
    repetition_penalty: float = 1.5    # 同じ文の繰り返し（モード崩壊）へのペナルティ
    num_beams: int = 10                # ビームサーチの探索幅
    diversity_penalty: float = 1.0     # 生成結果の多様性を生み出すためのペナルティ
    num_beam_groups: int = 10          # ビームサーチのグループ数
    num_return_sequences: int = 10     # 生成する文の数

    answer: str = "富士山"
    context: str = "富士山は静岡県と山梨県にまたがっている山です。"
    question: str = "静岡県と山梨県にはどんな山がありますか?"

    bad_words: list[str] = None

    verbose: bool = False               # デバグ用：引数を出力

    generate_type: str = "qa"   # 'qa' (Question Answering), AQG (Answer-aware Question Generation), AQG-HL (AQG with higlights)


def normalize_text(text):
    text = text.strip()
    assert "\t" not in text
    assert "\r" not in text
    assert "\n" not in text
    assert len(text) > 0

    text = text.replace("\u3000", " ").replace("\n", " ")
    text = mojimoji.zen_to_han(text, kana=False)  # 英数字を半角に
    text = mojimoji.han_to_zen(text, ascii=False, digit=False)  # かなを全角に

    text = neologdn.normalize(unicodedata.normalize('NFKC', text))
    text = text.lower() # t5-japaneseは英小文字 uncased らしい
    return text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
#    torch.manual_seed(seed)
#    if torch.cuda.is_available():
#        torch.cuda.manual_seed_all(seed)


def main():
    conf: Args = Args.from_args()
    if conf.verbose:
        print(conf)

    # answer = "富士山"
    # context = "富士山は静岡県と山梨県にまたがっている山です。"
    # question = "静岡県と山梨県にはどんな山がありますか?"
    answer = conf.answer
    context = conf.context
    question = conf.question

    # 前処理
    answer = normalize_text(answer.replace("\n", " "))
    context = normalize_text(context.replace("\n", " "))
    question = normalize_text(question.replace("\n", " "))

    # 乱数シード初期化
    set_seed(conf.seed)

    # プロンプト
    text2text_generaton_prompt = "question: {question} context: {context}"
    if conf.generate_type == 'aqg':
        text2text_generaton_prompt = "answer: {answer} context: {context}"
    if conf.generate_type == 'aqg-hl':
        text2text_generaton_prompt = "{context}"

    # モデルとパイプラインの初期化
    text2text_generator = pipeline("text2text-generation",
        model=conf.model_name_or_path,
        tokenizer=conf.tokenizer_name_or_path if conf.tokenizer_name_or_path is not None else conf.model_name_or_path)

    # 単純な禁句処理
    # 類義語や言い換え、一部の表記抜けや過剰な表記を考慮すべき。外部から指定させたほうがよい
    bad_words = conf.bad_words if conf.bad_words is not None else [answer]
    bad_words_ids = [text2text_generator.tokenizer(bad_word, add_special_tokens=False).input_ids[1:] for bad_word in bad_words]
    if len(bad_words_ids) == 0:
        bad_words_ids = None

    # 生成
    generated = text2text_generator(
                    text2text_generaton_prompt.format(answer=answer, question=question, context=context),
                    # 以下、 generate() への入力
                    max_length=conf.max_target_length ,
                    temperature=conf.temperature,
                    num_beams=conf.num_beams,
                    diversity_penalty=conf.diversity_penalty,
                    num_beam_groups=conf.num_beam_groups,
                    num_return_sequences=conf.num_return_sequences,
                    repetition_penalty=conf.repetition_penalty,
                    bad_words_ids=bad_words_ids
                )[0]['generated_text']

    print(generated)

if __name__ == "__main__":
    main()


"""
$ python generate.py --verbose  --generate_typ aqg
Args(model_name_or_path='sonoisa/t5-base-japanese-question-generation', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='富士山', context='富士山は静岡県と山梨県にまたがっている山です。', bad_words=None)
静岡県と山梨県にはどんな山がありますか?
$ python generate.py --verbose --answer 山梨県  --generate_typ aqg
Args(model_name_or_path='sonoisa/t5-base-japanese-question-generation', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='山梨県', context='富士山は静岡県と山梨県にまたがっている山です。', bad_words=None)
静岡県と他のどの県にまたがっている富士山ですか?
$ python generate.py --verbose --answer 山梨県 --context 山梨県の県庁所在地は甲府市です。  --generate_typ aqg
Args(model_name_or_path='sonoisa/t5-base-japanese-question-generation', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='山梨県', context='山梨県の県庁所在地は甲府市です。', bad_words=None)
どの県が県庁所在地ですか
$ python generate.py --verbose --answer 甲府市 --context 山梨県の県庁所在地は甲府市です。  --generate_typ aqg
Args(model_name_or_path='sonoisa/t5-base-japanese-question-generation', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='甲府市', context='山梨県の県庁所在地は甲府市です。', bad_words=None)
山梨県の県庁所在地はどこですか?
$ python generate.py --verbose --answer 甲府市 --context 山梨県の県庁所在地は甲府市です。 --bad_words 山梨 山梨県  --generate_typ aqg
Args(model_name_or_path='sonoisa/t5-base-japanese-question-generation', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='甲府市', context='山梨県の県庁所在地は甲府市です。', bad_words=['山梨', '山梨県'])
中部地方の県庁所在地はどこですか?
"""

"""
$ python generate.py --verbose --question 山梨県の県庁所在地は? --context 山梨県の県庁所在地は甲府市です。 --model_name_or_path ../../../t5-japanese/model_t5_JSQUAD_MRC --generate_typ qa
Args(model_name_or_path='../../../t5-japanese/model_t5_JSQUAD_MRC', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='富士山', context='山梨県の県庁所在地は甲府市です。', question='山梨県の県庁所在地は?', bad_words=None, verbose=True, generate_type='qa')
甲府市

$ python generate_question_answering.py --verbose --answer 甲府市 --question 山梨県の県庁所在地は? --context 山梨県の県庁所在地は甲府市です。 --model_name_or_path sonoisa/t5-base-japanese-question-generation --generate_typ aqg
Args(model_name_or_path='sonoisa/t5-base-japanese-question-generation', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='甲府市', context='山梨県の県庁所在地は甲府市です。', question='山梨県の県庁所在地は?', bad_words=None, verbose=True, generate_type='aqg')
山梨県の県庁所在地はどこですか?

$ python generate.py --verbose --answer 甲府市 --question 山梨県の県庁所在地は? --context 山梨県の県庁所在地は\<hl\>甲府市\<hl\>です。 --model_name_or_path ../../../t5-japanese/model_t5_JSQUAD_MRC --generate_typ aqg-hl
Args(model_name_or_path='../../../t5-japanese/model_t5_JSQUAD_MRC', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='甲府市', context='山梨県の県庁所在地は<hl>甲府市<hl>です。', question='山梨県の県庁所在地は?', bad_words=None, verbose=True, generate_type='aqg-hl')
甲府
"""