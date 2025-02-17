import os
import pandas as pd
from sqlalchemy import create_engine
from RecAlg import GazeData, PopulatioData, SearchTrendData, ExportAdvList
from util_data import *
import argparse
import json
import pymysql
from datetime import datetime, timedelta
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

pd.set_option('future.no_silent_downcasting', True)

def get_parser():
    parser = argparse.ArgumentParser(description="Generate CamelNeon Advertise Play List")
    parser.add_argument("--mode", type=bool, default=True, help='If you input predicted population (via LSTM) in this algorithm, insert True.')
    parser.add_argument("--loc", type=int, required=True, help='Input location number saved in database.')
    parser.add_argument("--start_date", type=str, default='2025-01-08', help='Input date format: YYYY-MM-DD (string)')
    # parser.add_argument("--end_date", type=str, default='2025-01-10 00:00:00')
    parser.add_argument("--search_option", type=bool, default=False, help='Use Naver search trend data')
    parser.add_argument("--DB_upload", type=bool, default=False, help='If you want upload exposure list to main DB, insert True in this argument')
    return parser

# MySQL 데이터베이스 연결
with open(f'./SQL_parameter.json', 'r') as file:
    par = json.load(file)

def insert_exposure_data_from_dataframe(df, loc_num):        
    connection = pymysql.connect(
        host=par['host'],
        user=par['username'],
        password=par['password'],
        database=par['database']
    )
    try:
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                cursor.execute("SELECT ad_id FROM ad_info WHERE title = %s", (row['name'],))
                result = cursor.fetchone()
                if result:
                    ad_id = result[0]
                    start_time = datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S')
                    end_time = start_time + timedelta(hours=1)
                    cursor.execute(
                        "INSERT INTO exposure (start_time, end_time, ad_id, region_id) VALUES (%s, %s, %s, %s)",
                        (start_time, end_time, ad_id, loc_num)
                    )
            connection.commit()
            print("Exposure data inserted successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        connection.rollback()
    finally:
        connection.close()

def main():
    os.makedirs('output', exist_ok=True)
    parser = get_parser()
    args = parser.parse_args()

    host = par['host']
    port = par['port']
    username = par['username']
    password = par['password']
    database = par['database']
    
    # 비밀번호 URL 인코딩
    encoded_password = urllib.parse.quote_plus(password)

    # SQLAlchemy 엔진 생성
    engine = create_engine(f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database}")
    ad_gaze, ad_info, ad_target, ad_images, frames, categories = import_data(engine)
    
    # 지역 선택
    loc_num = args.loc # [1:삼각지역, 2:군자역, 3:회기역, 4:'용산역]

    # 응시횟수 기반 광고 순서 결정
    df1 = GazeData.score_temp(ad_info, loc_num, ad_target)
    ad_gaze = GazeData.gaze_temp(ad_gaze, ad_info, frames)
    df_target = GazeData.calculate_score(ad_gaze, df1)
    keywords = list(df1['name'])
    df_gazed = GazeData.generate_add_actual(ad_gaze, keywords)
    gazed_df = GazeData.ranked_add_actual(df_gazed)
    
    pop = PopulatioData.import_data(engine, loc_num)
    pop, new_columns = PopulatioData.calculate_columns(pop)
    
    keywords = list(gazed_df.index)
    if args.search_option:
        naver_df = SearchTrendData.get_final_df(keywords)
        naver_df.to_csv('SearchTrend.csv')
    else:
        naver_df = pd.read_csv('SearchTrend.csv', index_col=0)
    
    if args.mode:
        final_exposure = export_exposure_using_lstm(gazed_df, naver_df, pop, args.start_date)
    else:
        final_exposure = export_exposure_only_data(gazed_df, naver_df, pop, args.start_date)
    final_exposure.to_csv(f'./output/final_exposure_{args.start_date.split()[0]}.csv', index=False)
    print('Exposure data generated successfully.')

    if args.DB_upload:
        insert_exposure_data_from_dataframe(final_exposure, loc_num)

if __name__ == "__main__":
    main()