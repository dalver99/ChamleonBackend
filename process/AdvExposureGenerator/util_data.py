import numpy as np
import pandas as pd
import urllib.parse
import ast
from RecAlg import ExportAdvList
import tensorflow as tf
from pickle import load
from datetime import datetime, timedelta

def create_datetime_list(start_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = start_date + timedelta(days=1)
    datetime_list = pd.date_range(start=start_date, end=end_date, freq="h")[:-1]
    return datetime_list

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
    
    query = "SELECT * FROM ad_images"
    ad_images = pd.read_sql(query, engine)
    
    query = "SELECT * FROM frame"
    frames = pd.read_sql(query, engine)
    
    query = "SELECT * FROM product_category"
    categories = pd.read_sql(query, engine)   
    
    return ad_gaze, ad_info, ad_target, ad_images, frames, categories

def edit_data(ad_info, ad_gaze, target):
    ad_id = ad_info[ad_info['title']==target]['ad_id'].iloc[0]
    ad_gt = ad_gaze[ad_gaze['ad_id']==ad_id].copy() #ad_gt: ad_gaze_target
    ad_gt['gaze_num'] = [i for i in range(1, len(ad_gt)+1)]
    ad_gt.reset_index(drop='index', inplace=True)
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

def export_exposure_only_data(gazed_df, naver_df, pop, st_date):
    final_exposure = pd.DataFrame(columns=['name', 'time'])  # 'time' 열 추가
    datetime_list, end_date = create_datetime_list(st_date)
    perv_adv = []
    ad_num = 1
    for dt in datetime_list:
        if dt.hour == 0:
            perv_adv = []   
        try:
            # 광고 랭킹 점수 계산
            final_rank = ExportAdvList.get_rank_score(gazed_df, naver_df, pop, dt)
            
            # 새로운 광고 리스트 생성
            new_rows = pd.DataFrame({'name': list(final_rank.index), 'time': dt})  # 광고 이름과 시간 추가
            
            # 이전에 선택된 광고 제외
            new_rows = new_rows.loc[~(new_rows['name'].isin(perv_adv))]
            #print(new_rows)
            # 상위 ad_num개 광고를 결과에 추가
            final_exposure = pd.concat([final_exposure, new_rows.iloc[:ad_num]], ignore_index=True)
            
            # 중복 방지를 위해 선택된 광고를 저장
            perv_adv.extend(new_rows['name'][:ad_num])
            
            # 광고가 한바퀴 돌았으면, perv_adv 리셋
            if len(perv_adv) == len(final_rank):
                perv_adv = []  # 정시에 리셋
            
        except Exception as e:
            #print(f"Error processing time {dt}: {str(e)}")
            continue

    # 결과 출력
    #display(final_exposure)
    return final_exposure

def using_lstm(pop, datetime_list):
    new_columns = ['총생활인구수', '10gen_male', '10gen_female', '20gen_male', '20gen_female',
                        '30gen_male', '30gen_female', '40gen_male', '40gen_female',
                        '50gen_male', '50gen_female', '60gen_male', '60gen_female']
    
    pop_lstm = pop.copy()
    pop_lstm['총생활인구수'] = pop_lstm.sum(axis=1)

    # 5분 단위의 새로운 시계열 인덱스 생성
    new_index = pd.date_range(start=pop_lstm.index.min(), 
                            end=pop_lstm.index.max(), 
                            freq='5min')

    # 새로운 인덱스로 리샘플링하고 선형 보간
    pop_lstm = pop_lstm.reindex(new_index).interpolate(method='linear')
    pop_lstm = pop_lstm.loc[datetime_list]
    pop_lstm = pop_lstm[new_columns]
    
    # 기존의 모델 및 scaler 가져오기
    scaler = load(open('./LSTM_model/minmax_scaler.pkl', 'rb'))
    input_scaled = scaler.transform(pop_lstm)
    input_value = input_scaled[:24,:]
    input_value = input_value.reshape((-1, input_value.shape[0], input_value.shape[1]))
    model = tf.keras.models.load_model('./LSTM_model/best_model.h5', 
                                    custom_objects={'mae': tf.keras.metrics.mae})
    
    # 훈련시킨 LSTM 모델 적용
    # model.summary()
    pop_pred = model.predict(input_value)

    # 3D 배열을 2D로 재구성
    pop_pred_2d = pop_pred.reshape(-1, pop_pred.shape[-1])

    # 스케일링 역변환 적용
    pop_pred_rescaled = scaler.inverse_transform(pop_pred_2d)

    # 결과를 다시 3D로 재구성 (필요한 경우)
    pop_pred = pop_pred_rescaled.reshape(pop_pred.shape)
    
    return pop_pred, new_columns

def create_datetime_list(start_date_str):
    # 시작 날짜를 datetime 객체로 변환
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    
    # 종료 날짜를 시작 날짜의 다음 날로 설정
    end_date = start_date + timedelta(days=1)
    
    # datetime 리스트 생성
    datetime_list = pd.date_range(start=start_date, end=end_date, freq="h")[:-1]
    
    return datetime_list, end_date

# Section1: 해당 시간 ~ 1시간 후까지 효과 좋은 광고 리스트 뽑는 코드
def export_exposure_using_lstm(gazed_df, naver_df, pop, st_date):
    # Final_exposure 빈 temp 생성
    final_exposure = pd.DataFrame(columns=['name', 'time'])  # 'time' 열 추가
    perv_adv = []
    ad_num = 1
    
    # LSTM으로 st_date + 24시간의 인구수 예측
    datetime_list, end_date = create_datetime_list(st_date)  # 00시부터 23시까지 생성
    pop_pred, new_columns = using_lstm(pop, datetime_list)
    pop_pred = pop_pred.squeeze(axis=0)
    new_date = end_date + timedelta(days=1)
    new_datetime_list = pd.date_range(start=end_date, end=new_date, freq="h")[:-1]
    pop_final = pd.DataFrame(pop_pred, index=new_datetime_list, columns=new_columns)
    pop_final = pop_final.loc[:, new_columns[1:]]
    
    for dt in new_datetime_list:
        if dt.hour == 0:
            perv_adv = []   
        try:
            # 광고 랭킹 점수 계산
            final_rank = ExportAdvList.get_rank_score(gazed_df, naver_df, pop_final, dt)
            
            # 새로운 광고 리스트 생성
            new_rows = pd.DataFrame({'name': list(final_rank.index), 'time': dt})  # 광고 이름과 시간 추가
            
            # 이전에 선택된 광고 제외
            new_rows = new_rows.loc[~(new_rows['name'].isin(perv_adv))]
            #print(new_rows)
            # 상위 ad_num개 광고를 결과에 추가
            final_exposure = pd.concat([final_exposure, new_rows.iloc[:ad_num]], ignore_index=True)
            
            # 중복 방지를 위해 선택된 광고를 저장
            perv_adv.extend(new_rows['name'][:ad_num])
            
            # 광고가 한바퀴 돌았으면, perv_adv 리셋
            if len(perv_adv) == len(final_rank):
                perv_adv = []  # 정시에 리셋
            
        except Exception as e:
            print(f"Error processing time {dt}: {str(e)}")
            continue

    # 결과 출력
    #display(final_exposure)
    return final_exposure

