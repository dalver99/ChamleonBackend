import numpy as np
import pandas as pd
import urllib.parse
import ast

def import_data(engine):
    # import ad_gaze data from sql
    query = "SELECT * FROM ad_gaze"
    ad_gaze = pd.read_sql(query, engine)
    bins = [0, 20, 30, 40, 50, 60, float('inf')]
    labels = ['10gen', '20gen', '30gen', '40gen', '50gen', '60gen']
    ad_gaze['age_group'] = pd.cut(ad_gaze['pre_age'], 
                            bins=bins, 
                            labels=labels, 
                            right=False)
    
    # import ad_info dataset
    query = "SELECT * FROM ad_info"
    ad_info = pd.read_sql(query, engine)
    
    # import ad_target dataset
    query = "SELECT * FROM target"
    ad_target = pd.read_sql(query, engine)
    ad_target['target_info'] = ad_target['target_info'].apply(ast.literal_eval)
    
    query = "SELECT * FROM frame"
    frame = pd.read_sql(query, engine)
    return ad_gaze, ad_info, ad_target, frame

def edit_data(ad_info, ad_gaze, frame, target):
    ad_id = int(ad_info[ad_info['title']==target]['ad_id'].iloc[0])
    frame_ids = frame[frame['ad_id']==ad_id]['id'].to_list()
    ad_gt = ad_gaze[ad_gaze['frame_id'].isin(frame_ids)].copy() #ad_gt: ad_gaze_target
    ad_gt.reset_index(drop='index', inplace=True)
    ad_gt['gazed_num'] =  ad_gt['gazed'].cumsum()
    return ad_id, ad_gt

def generate_input(ad_gt):
    labels = ['male', 'female', '10gen', '20gen', '30gen', '40gen', '50gen', '60gen', 'gaze_num']
    data_pre = pd.DataFrame(columns=labels)
    data_pre['gaze_num'] = ad_gt['gaze_num']
    data_pre.fillna(0, inplace=True)
    data_pre.reset_index(drop='index', inplace=True)
    #Count Gender
    for idx, row in ad_gt.iterrows():
        gender = row['pre_gender']
        group = row['age_group']
        data_pre.loc[idx, gender] = data_pre[gender].sum() + 1
        data_pre.loc[idx, group] = data_pre[group].sum() + 1
    return data_pre

# 재추천 결과 출력을 위한 문자열 변환 함수
def format_age_groups(ages):
    return [age.replace('gen', '대') for age in ages if 'gen' in age]

def format_gender(list):
    if 'male' in list:
        return '남성'
    elif 'female' in list:
        return '여성'
    elif 'all' in list:
        return '모두'
    if ('male' in list) and ('female' in list):
        return '모두'
    if ('male' not in list) and ('female' not in list):
        return '모두'

def export_targets(ad_target, ad_id, rt_df):
    raw_target = ad_target[ad_target['id']==ad_id]['target_info'].iloc[0]
    recommand_target = list(rt_df.index)
    return raw_target, recommand_target
