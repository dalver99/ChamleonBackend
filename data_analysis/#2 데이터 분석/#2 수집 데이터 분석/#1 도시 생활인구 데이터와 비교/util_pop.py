import pandas as pd
import numpy as numpy
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
import json
from sqlalchemy import create_engine
import urllib.parse
from sklearn.metrics import r2_score

def import_db_data(engine, loc_num, option=False):
    # import adv_gaze data from sql
    query = f"SELECT * FROM population"
    if option == True:
        query = f"SELECT * FROM population WHERE region_id = {loc_num}"
    pop =pd.read_sql(query, engine)
    return pop

def calculate_columns(df):
    generation_columns = ['10gen', '20gen', '30gen', '40gen', '50gen', '60gen']
    new_columns = ['datetime', 'region_id', 'max_population']
    for gen in generation_columns:
        df[f"{gen}_male_db"] = df['male_rate'] / 100 * df['max_population'] * df[gen] / 100
        df[f"{gen}_female_db"] = df['female_rate'] / 100 * df['max_population'] * df[gen] / 100
        new_columns.append(f"{gen}_male_db")
        new_columns.append(f"{gen}_female_db")
    return df[new_columns], new_columns

def import_real_data(fn, targets):
    data = pd.read_csv(fn)
    data = data[data['AREA_NM'].isin(targets)]
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y%m%d%H%M%S')
    columns = ['10gen_male', '10gen_female', '20gen_male', 
               '20gen_female', '30gen_male', '30gen_female', 
               '40gen_male', '40gen_female',  '50gen_male', 
               '50gen_female', '60gen_male', '60gen_female']
    new_columns = {}
    for col in columns:
        new_columns[col] = f'{col}_lp'
    data.rename(columns=new_columns, inplace=True)
    return data

def import_df_columns():
    selected_cols = ['datetime', 'max_population', '총생활인구수']
    for i in range(10, 70, 10):
        for g in ['male', 'female']:
            for t in ['db','lp']:
                selected_cols.append(f'{i}gen_{g}_{t}')
    selected_cols.append('AREA_NM_x')
    return selected_cols

def edit_final_data(db_pop, lp_pop, selected_cols, target_areas):
    final_df = pd.DataFrame()
    for target_area in target_areas:
        db_target = db_pop[db_pop['AREA_NM']==target_area]
        lp_target = lp_pop[lp_pop['AREA_NM']==target_area]
        merged_data = pd.merge(db_target, lp_target, left_on='datetime', right_on='Datetime', how='inner')
        merged_data = merged_data[selected_cols]
        merged_data.set_index('datetime', inplace=True)
        final_df = pd.concat([final_df, merged_data])
    return final_df

def compare_pop(final_df):
    # 데이터 필터링
    filtered_data = final_df.drop(columns='AREA_NM_x')
    plt.figure(figsize=(6, 4))
    plt.subplots_adjust(right=1)
    # 시각화를 위한 데이터 준비
    female_list =[]
    male_list =[]
    weight = 1
    for i in range(10, 70, 10):
        for g in ['male', 'female']:
            male_data = filtered_data[[f'{i}gen_{g}_db', f'{i}gen_{g}_lp']]
            if g == 'male':
                male_list.append(r2_score(male_data[f'{i}gen_{g}_db'], male_data[f'{i}gen_{g}_lp'] * weight))
            else:
                female_list.append(r2_score(male_data[f'{i}gen_{g}_db'], male_data[f'{i}gen_{g}_lp'] * weight))
        # 그래프 설정
            plt.scatter(male_data[f'{i}gen_{g}_db'], male_data[f'{i}gen_{g}_lp'] * weight, label=f'{i}gen_{g}')
    plt.xlabel('실시간 인구수 데이터')
    plt.ylabel('추정 인구수')
    plt.legend(bbox_to_anchor=(1.05, 1), 
        loc='upper left')

    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(index=[f'{i}_gen' for i in range(10, 70, 10)], columns=['male', 'female'])
    df['male'] = male_list
    df['female'] = female_list
    return df

def corr_bar(final_df, detail_option, target):
    target_df = final_df.copy()
    bar_df = target_df.corr(numeric_only=True)
    age_cols = [f'{i}gen' for i in range(10, 70, 10)]
    gen_cols = ['male', 'female']
    x_data = []
    y_data = []
    plt.figure(figsize=(6,4))
    for age in age_cols:
        for gen in gen_cols:
            sel_cols = [f'{age}_{gen}_db', f'{age}_{gen}_lp']
            v = bar_df.loc[sel_cols[0], sel_cols[1]]
            y_data.append(v)
            x_data.append(f'{age}_{gen}')
    plt.bar(x=x_data, height=y_data)
    plt.xticks(rotation=45)
    plt.ylabel('Corr.')
    plt.ylim([0,1])
    if detail_option == True:
        plt.title(f'{target}')
    plt.show()

def analyze_correlations(df, option, target='전체 지'):
    if option == 1:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(numeric_only=True), 
                    annot=True,
                    cmap='RdBu_r',
                    vmin=0, vmax=1,
                    fmt='.2f')
    
    if option == 2:
        cols = list(df.columns)
        for i in range(0, len(cols), 2):
            sel_cols = cols[i:i+2]
            correlation_matrix = df[sel_cols].corr()
        
            plt.figure(figsize=(4, 4))
            sns.heatmap(correlation_matrix, 
                        annot=True,
                        cmap='RdBu_r',
                        vmin=0, vmax=1,
                        fmt='.2f')
    
    plt.title(f'{target}역 실시간 인구 수 - 생활 인구 수 데이터의 연령/성별 간 상관 계수')
    plt.tight_layout()
    plt.show()