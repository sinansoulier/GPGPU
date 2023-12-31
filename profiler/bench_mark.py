import pandas as pd
import numpy as np
import re
import argparse
import matplotlib.pyplot as plt
import os
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-f','--file', type=str, default='analyse.csv',
                    help='file name')
parser.add_argument('-o','--output-file', type=str, default='analyse.csv',
                    help='output file name')
csv_file = parser.parse_args().file
output_file = parser.parse_args().output_file

if csv_file is None:
    csv_file = 'analyse.csv'
if output_file is None:
    output_file = 'clean_benchmark.csv'
    
if not os.path.exists(csv_file):
    print('File not found')
    exit(1)

def generate_clean_benchmark(csv_file):
    bench_mark_df = pd.read_csv(csv_file)
    bench_mark_df.iloc[0] = bench_mark_df.iloc[0].fillna('')
    for col in bench_mark_df.columns:
        if bench_mark_df.loc[0,col] != '':
            bench_mark_df.rename(columns={col: f'{col} ({bench_mark_df.loc[0,col]})'}, inplace=True)
    bench_mark_df = bench_mark_df.drop([0])
    bench_mark_df
    bench_mark_df = bench_mark_df.T
    bench_mark_df.columns = bench_mark_df.loc['Name']
    bench_mark_df = bench_mark_df.drop(['Name'])
    function_regex = re.compile(r'([\w_]+)\(')
    function_column = []
    col_to_keep = ['[CUDA memcpy DtoH]',  '[CUDA memcpy HtoD]' ]
    for col in bench_mark_df.columns:
        
        matches = function_regex.findall(col)
        if len(matches) > 0:
            function_column.append(matches[0])
            bench_mark_df.rename(columns={col: matches[0]}, inplace=True)
        elif col in col_to_keep:
            function_column.append(col)
    bench_mark_df[function_column] 
    bench_mark_df = bench_mark_df[function_column]
    bench_mark_df.loc['Type']
    bench_mark_df = bench_mark_df.drop(['Type'])
    bench_mark_df = bench_mark_df.astype(float)
    bench_mark_df.to_csv('clean_benchmark.csv')
    fig, ax = plt.subplots(figsize=(20, 10))
    transposed_df = (bench_mark_df.drop(['Calls']).T)

    transposed_df.plot(ax=ax, kind='bar',  rot=0)
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Function')
    fig.tight_layout()
    fig.savefig('benchmark1.png')
    return bench_mark_df



df = generate_clean_benchmark(csv_file)
print(df)
