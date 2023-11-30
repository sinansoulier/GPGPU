import pandas as pd
import numpy as np
import re
import argparse
import matplotlib.pyplot as plt
import os
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file', type=str, default='analyse.csv',
                    help='file name')
parser.add_argument('--output-file', type=str, default='clean_benchmark.csv',
                    help='output file name')
csv_file = parser.parse_args().file
output_file = parser.parse_args().output_file

if csv_file is None:
    # show usage
    parser.print_help()
    csv_file = 'analyse.csv'
if output_file is None:
    output_file = 'clean_benchmark.csv'
    
if not os.path.exists(csv_file):
    print('File not found')
    exit(1)

def generate_clean_benchmark(csv_file):
    bench_mark_df = pd.read_csv(csv_file)
    bench_mark_df = bench_mark_df.T
    bench_mark_df.columns = bench_mark_df.loc['Name']
    bench_mark_df = bench_mark_df.drop(['Name'])
    function_regex = re.compile(r'([\w_]+)\(')
    function_column = []
    for col in bench_mark_df.columns:
        matches = function_regex.findall(col)
        if len(matches) > 0:
            function_column.append(matches[0])
            bench_mark_df.rename(columns={col: matches[0]}, inplace=True)
    bench_mark_df[function_column] 
    df = bench_mark_df[function_column]
    df.loc['Type']
    df = df.drop(['Type'])
    df = df.astype(float)
    df.to_csv('clean_benchmark.csv')
    df = df.T
    fig, ax = plt.subplots(figsize=(20, 10))
    df.plot(ax=ax, kind='bar')
    
    ax.set_xlabel('Function')
    ax.set_title('Benchmark')
    return df

generate_clean_benchmark(csv_file)
