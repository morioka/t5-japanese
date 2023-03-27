import pandas as pd
import os

dir_in = "data_jsquad_aqg_hl"
dir_out = "data_jsquad_aqg_hl_gpt"

assert dir_in != dir_out

os.makedirs(dir_out, exist_ok=True)

for data_file in ['train', 'dev', 'test']:
    df = pd.read_csv(f'{dir_in}/{data_file}.tsv', sep='\t')

    with open(f'{dir_out}/{data_file}.txt', 'w') as f:
        for _, row in df.iterrows():
            _, i, o = row
            f.write(f'<s>{i}[SEP]{o}</s>\n')


##

dir_in = "data_jsquad_aqg"
dir_out = "data_jsquad_aqg_gpt"

assert dir_in != dir_out

os.makedirs(dir_out, exist_ok=True)

for data_file in ['train', 'dev', 'test']:
    df = pd.read_csv(f'{dir_in}/{data_file}.tsv', sep='\t')

    with open(f'{dir_out}/{data_file}.txt', 'w') as f:
        for _, row in df.iterrows():
            _, i, o = row
            f.write(f'<s>{i}[SEP]{o}</s>\n')
