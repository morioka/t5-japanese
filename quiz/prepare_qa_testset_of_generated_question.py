
#
#   AQGが出力した questionで、QA評価用のquestionを差し替える。
#
# create_qa_testset_of_genereated_question.py


import pandas as pd

#GENERATED_OUTPUT='model/test_output.tsv'
#DATA_TEST_FILE='data/test.tsv'

GENERATED_OUTPUT='output.tsv'   # qa_id	input(answer: {answer_text} context: {context})	target(question)	output(question)
DATA_TEST_FILE='data/test.tsv'  # qa_id 'question: {question} context: {context}' answer_text

df_generated = pd.read_csv(GENERATED_OUTPUT, sep='\t')
# qa_id	input	target	output
df_test = pd.read_csv(DATA_TEST_FILE, sep='\t', header=None)


col_qa_id = df_test.iloc[:, 0]
col_question = df_test.iloc[:, 1]

for i, row in df_test.iterrows():
    qa_id, question_and_context, answer_text = row

    try:
        question = df_generated[df_generated.qa_id == qa_id]['output'].values[0]

        prompt_context = 'context: '
        context = question_and_context[question_and_context.find(prompt_context) + len(prompt_context):]

        new_question_and_context = f"question: {question} context: {context}"

        #df_test[df_test.iloc[:, 0] == qa_id].iloc[:, 1] = new_question_and_context
        row[1] = new_question_and_context
    except:
        print('error: ', row)
        pass

df_test.to_csv('test_generated.tsv', sep='\t', index=None, header=None)
