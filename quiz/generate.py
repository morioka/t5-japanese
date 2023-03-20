# -*- coding: utf-8 -*-

import random
import numpy as np

from transformers import pipeline

import unicodedata
import neologdn

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

    bad_words: list[str] = None

    verbose: bool = False               # デバグ用：引数を出力


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
    conf: Args = Args.from_args()
    if conf.verbose:
        print(conf)

    # answer = "富士山"
    # context = "富士山は静岡県と山梨県にまたがっている山です。"
    # question = "静岡県と山梨県にはどんな山がありますか?"
    answer = conf.answer
    context = conf.context

    # 前処理
    answer = normalize_text(answer.replace("\n", " "))
    context = normalize_text(context.replace("\n", " "))

    # 乱数シード初期化
    set_seed(conf.seed)

    # プロンプト
    text2text_question_generaton_prompt = "answer: {answer} context: {context}"

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
                    text2text_question_generaton_prompt.format(answer=answer, context=context),
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
(t5-jp) morioka@legion:~/aio/morioka/t5-japanese/quiz$ python ./predict.py
Args(model_name_or_path='sonoisa/t5-base-japanese-question-generation', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='富士山', context='富士山は静岡県と山梨県にまたがっている山です。', bad_words=None)
静岡県と山梨県にはどんな山がありますか?
(t5-jp) morioka@legion:~/aio/morioka/t5-japanese/quiz$ python ./predict.py --answer 山梨県
Args(model_name_or_path='sonoisa/t5-base-japanese-question-generation', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='山梨県', context='富士山は静岡県と山梨県にまたがっている山です。', bad_words=None)
静岡県と他のどの県にまたがっている富士山ですか?
(t5-jp) morioka@legion:~/aio/morioka/t5-japanese/quiz$ python ./predict.py --answer 山梨県 --context 山梨県の県庁所在地は甲府市です。
Args(model_name_or_path='sonoisa/t5-base-japanese-question-generation', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='山梨県', context='山梨県の県庁所在地は甲府市です。', bad_words=None)
どの県が県庁所在地ですか
(t5-jp) morioka@legion:~/aio/morioka/t5-japanese/quiz$ python ./predict.py --answer 甲府市 --context 山梨県の県庁所在地は甲府市です。
Args(model_name_or_path='sonoisa/t5-base-japanese-question-generation', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='甲府市', context='山梨県の県庁所在地は甲府市です。', bad_words=None)
山梨県の県庁所在地はどこですか?
(t5-jp) morioka@legion:~/aio/morioka/t5-japanese/quiz$ python ./predict.py --answer 甲府市 --context 山梨県の県庁所在地は甲府市です。 --bad_words 山梨 山梨県
Args(model_name_or_path='sonoisa/t5-base-japanese-question-generation', tokenizer_name_or_path=None, seed=42, max_input_length=512, max_target_length=64, temperature=1.0, repetition_penalty=1.5, num_beams=10, diversity_penalty=1.0, num_beam_groups=10, num_return_sequences=10, answer='甲府市', context='山梨県の県庁所在地は甲府市です。', bad_words=['山梨', '山梨県'])
中部地方の県庁所在地はどこですか?
(t5-jp) morioka@legion:~/aio/morioka/t5-japanese/quiz$ """
