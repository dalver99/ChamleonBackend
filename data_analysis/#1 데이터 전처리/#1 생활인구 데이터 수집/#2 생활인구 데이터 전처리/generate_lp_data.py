import pandas as pd
import argparse
import os
from tqdm import tqdm

def data_format():
    dtype_dict = {'기준일ID': 'int32', '시간대구분': 'int32',
                  '행정동코드': 'int32', '집계구코드': 'int64',
                  '총생활인구수': 'float32', '10gen_male': 'float32', 
                  '10gen_female': 'float32', '20gen_male': 'float32',
                  '20gen_female': 'float32', '30gen_male': 'float32',
                  '30gen_female': 'float32', '40gen_male': 'float32', 
                  '40gen_female': 'float32', '50gen_male': 'float32', 
                  '50gen_female': 'float32', '60gen_male': 'float32', '60gen_female': 'float32'}
    return dtype_dict

def get_parser():
    parser = argparse.ArgumentParser(description="Please input target location")
    parser.add_argument("--target", type=str, default=None, required=True, help="target location")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    target = args.target
    dtype_spec = data_format()
    path = './생활인구데이터/'
    fn_list = os.listdir(path)
    region_code = pd.read_csv('Rate.csv')
    cols = ['TOT_REG_CD', 'AREA_NM', 'ADM_CD', 'rate']
    region_code_edit = region_code.loc[region_code['AREA_NM']==target]
    rcl = region_code_edit['TOT_REG_CD'].unique()
    rcl = [int(i) for i in rcl]
    
    df = pd.DataFrame()
    for fn in tqdm(fn_list):
        df_ = pd.read_csv(path + fn, dtype=dtype_spec, index_col=0)
        df = pd.concat([df, df_], ignore_index = True)
    df['Datetime'] = pd.to_datetime(df['기준일ID'].astype(str), format='%Y%m%d')
    df['Datetime'] = df['Datetime'] + pd.to_timedelta(df['시간대구분'], unit='h')
    df.set_index('Datetime', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    merged_df = pd.merge(df, region_code_edit, how='left', left_on='집계구코드', right_on='TOT_REG_CD')
    merged_df.set_index('Datetime', inplace=True)
    merged_df.sort_index(ascending=True, inplace=True)
    gen_columns = [col for col in merged_df.columns if 'gen' in col]
    for col in gen_columns:
        merged_df[f'{col}'] = merged_df[col] * merged_df['rate']
    cols = ['총생활인구수']
    for i in range(10,70,10):
        name = f'{i}gen'
        cols.append(f'{name}_male')
        cols.append(f'{name}_female')
    final_df = merged_df.groupby(['Datetime'])[cols].sum()
    final_df.to_csv(f'{target}_생활인구수.csv', encoding='utf-8')
    print('finish')

if __name__ == '__main__':
    main()