import os
import sys
import urllib.request
import datetime
import json
import math
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import mysql.connector
from mysql.connector import Error
os.makedirs('output', exist_ok=True)

class GazeData():
    def score_temp(ad_info, loc_num, ad_target):
        ad_info = ad_info[ad_info['region_id'] == loc_num].copy() # 설정된 타겟이 실제로 해당 광고를 봤는지 검토하기 위한 DF
        col = ['name','location', 'gender', '10gen', '20gen', '30gen',	'40gen', '50gen', '60gen', 'category', 'age_weight', 'gender_weight', 'raw_score', 'score']
        df = pd.DataFrame(columns=col)
        df['name'] = ad_info['title']
        df['location'] = 1
        list_ = ad_info['target_id'].to_list()
        result = list(pd.Series(list_).map(ad_target.set_index('id')['target_info']))
        df['gender'] = [r[0] for r in result]

        # 방법 2: apply 함수 사용 (더 pandas스러운 방법)
        def map_values(row_idx):
            row = df.iloc[row_idx]
            for col in result[row_idx]:
                if col in df.columns:
                    row[col] = True
            return row

        df = df.apply(lambda x: map_values(df.index.get_loc(x.name)), axis=1)
        names = df['name'].index
        df['category'] = ad_info.loc[names, 'target_id']
        df[['10gen', '20gen', '30gen',	'40gen', '50gen', '60gen']] = df[['10gen', '20gen', '30gen', '40gen', '50gen', '60gen']].fillna(False).astype(bool)
        df['gender_weight'] = [2 if i == 'all' else 1 for i in df['gender']]
        df['age_weight'] = df[['10gen', '20gen', '30gen', '40gen', '50gen', '60gen']].sum(axis=1)
        return df

    def gaze_temp(ad_gaze, ad_info, frames):
        df2 = frames[['id', 'ad_id', 'captured_at']]
        df2.columns = ['frame_id', 'ad_id', 'captured_at']
        merged_df = pd.merge(ad_gaze, df2, on='frame_id')
        ad_gaze = merged_df[['ad_id', 'pre_age', 'pre_gender', 'gazed', 'age_group', 'captured_at']]
        ad_gaze.columns = ['ad_id', 'pre_age', 'gender', 'gazed', 'age', 'captured_at']
        ad_gaze = pd.merge(ad_gaze, ad_info[['ad_id', 'title']], on='ad_id')
        return ad_gaze

    def calculate_score(ad_gaze, df):
        gazed_title = list(ad_gaze['title'].unique())
        df_target = df[df['name'].isin(gazed_title)]
        df_target.set_index('name', inplace=True)
        keywords = df_target.index
        
        for name in keywords:
            label = [f'{i}0gen' for i in range(2, 7)]
            df_temp = df_target.loc[name]
            gender = df_temp.loc['gender']
            if gender == 'all':
                gender_cols = ['female', 'male']
            else:
                gender_cols = [gender]

            true_cols = df_temp[df_temp==True].index
            age_cols = [i for i in true_cols if i in label]
            idx = list(ad_gaze[(ad_gaze['gazed']==True) & 
                            (ad_gaze['age'].isin(age_cols)) & 
                            (ad_gaze['gender'].isin(gender_cols))].index)
            df_target.loc[name, 'raw_score'] = len(idx)
            df_target.loc[name, 'score'] = df_target.loc[name, 'raw_score']/(df_target.loc[name, 'age_weight']*df_target.loc[name, 'gender_weight'])
        return df_target

    def generate_add_actual(cv_raw, keywords):
        # Columns 생성
        age_cols = ['10gen', '20gen', '30gen', '40gen', '50gen', '60gen']
        gender_cols = ['male', 'female']
        df_cols = ['name']
        for age in age_cols:
            for gender in gender_cols:
                df_cols.append(f'{age}_{gender}')
        
        # DF 생성
        df = pd.DataFrame(columns=df_cols)
        df['name'] = keywords
        df.set_index('name', inplace=True)
        
        for name, group in cv_raw.groupby(['title', 'age', 'gender'], observed=True):
            if f"{name[1]}_{name[2]}" in df.columns:
                df.loc[name[0], f"{name[1]}_{name[2]}"] = len(group)

        df = df.fillna(0)
        return df

    def ranked_add_actual(df):
        res_df = pd.DataFrame(index=df.index)
        cols = df.columns
        for col in cols:
            res_df[f'{col}'] = df[col].rank(ascending=False, method='min')
        return res_df

class PopulatioData():
    def calculate_columns(df):
        generation_columns = ['10gen', '20gen', '30gen', '40gen', '50gen', '60gen']
        new_columns = []
        for gen in generation_columns:
            df[f"{gen}_male"] = df['male_rate'] / 100 * df['max_population'] * df[gen] / 100
            df[f"{gen}_female"] = df['female_rate'] / 100 * df['max_population'] * df[gen] / 100
            new_columns.append(f"{gen}_male")
            new_columns.append(f"{gen}_female")
        return df[new_columns], new_columns

    def import_data(engine, loc_num):
        # import adv_gaze data from sql
        query = f"SELECT * FROM population WHERE region_id = {loc_num}"
        pop =pd.read_sql(query, engine)
        pop.set_index('datetime', inplace=True)
        return pop
    
    def select_time(df, datetime):
        return df.loc[datetime]
        
class ExportAdvList():
    def get_rank_score(gaze_df, rank_df, population_df, time):  
        """
        gaze_df: 응시 랭킹 df
        rank_df: 검색량 랭킹 df
        population_df: 유동 인구 랭킹 df

        응시 랭킹, 검색량 랭킹과 유동 인구 랭킹을 비교하여 점수를 계산한 df
        """
        res_df = pd.DataFrame(index=rank_df.index)
        cols = rank_df.columns  # 'keyword' 칼럼을 제외한 나머지 칼럼들

        # 유동 인구 데이터에 대해 랭킹을 계산
        for col in cols:
            # 유동 인구 랭킹을 계산 (랭킹이 낮을수록 좋음)
            population_df[f'{col}_rank'] = population_df[col].rank(ascending=False, method='min')
            
        scores = []
        for name in rank_df.index:
            score = 0
            for col in cols:
                gaze_rank = gaze_df[f'{col}'].loc[name]
                search_rank = rank_df[f'{col}'].loc[name]
                pop_sample_rank = population_df.iloc[:,:12]
                pop_rank = pop_sample_rank.loc[time, col]  # 특정 열을 선택

                # 각 순위가 스칼라 값인지 확인하고, 시리즈인 경우 첫 번째 값을 사용
                gaze_rank = gaze_rank.iloc[0] if isinstance(gaze_rank, pd.Series) else gaze_rank
                search_rank = search_rank.iloc[0] if isinstance(search_rank, pd.Series) else search_rank
                pop_rank = pop_rank.iloc[0] if isinstance(pop_rank, pd.Series) else pop_rank

                score += abs(gaze_rank + search_rank + pop_rank)
            scores.append(score)

        # 계산된 점수를 결과 데이터프레임에 추가
        res_df['score'] = scores

        # 이 점수들에 대해 가장 작은 점수를 100점, 가장 안좋은 점수를 0점으로 잡아 점수화
        min_score = min(score for score in scores if not isinstance(score, pd.Series))
        max_score = max(score for score in scores if not isinstance(score, pd.Series))

        res_df['normalized_score'] = res_df['score'].apply(lambda x: (max_score - x) / (max_score - min_score) * 100)
        res_df.sort_values('normalized_score', ascending=False, inplace=True)
        return res_df    

class SearchTrendData():
    def get_api_data(keywords, gender, ages):
        """
        네이버 데이터랩 API를 호출하여 검색 데이터를 가져오는 함수.

        Args:
            keywords: 광고명 검색 키워드 array, 최대 5개. 총 검색 광고가 5개 이상일 경우 항상 첫번째 광고를 입력해 줘야 스케일링이 가능해짐
            gender: 남여, m/f
            ages: 연령대. ~/12/18/24/29/34/39/44/49/54/59/~ 범위로 1~11 사이의 정수

        Returns:
            dict: API 응답 JSON 데이터를 파싱한 딕셔너리 반환. 누락된 날짜의 ratio 값은 0으로 채워지고, 첫 번째 키워드 기준으로 정규화됨.
        """
        # 오늘 날짜 기준 일주일 전부터 데이터를 가져오기
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=1)
        keyword_groups = [{"groupName": keyword, "keywords": [keyword]} for keyword in keywords]
        
        # 요청 데이터 구성
        body = json.dumps({
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "timeUnit": "date",
            "keywordGroups": keyword_groups,
            "ages": ages,
            "gender": gender
        })
        
        with open(f'naver_api_key.json', 'r') as file:
            keys = json.load(file)
            
        # API 요청 헤더 설정 -> 향후 txt 파일에서 load 해서 가져오는 방식으로 변경 요망
        request = urllib.request.Request(keys['url'])
        request.add_header("X-Naver-Client-Id", keys['client_id'])
        request.add_header("X-Naver-Client-Secret", keys['client_secret'])
        request.add_header("Content-Type", "application/json")

        try:
            # 요청 실행
            response = urllib.request.urlopen(request, data=body.encode("utf-8"))
            rescode = response.getcode()

            if rescode == 200:
                response_body = response.read()
                api_data = json.loads(response_body.decode("utf-8"))

                # 누락된 날짜의 ratio를 0으로 채우기
                all_dates = [(start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
                for result in api_data['results']:
                    existing_dates = {entry['period']: entry['ratio'] for entry in result['data']}
                    result['data'] = [{"period": date, "ratio": existing_dates.get(date, 0)} for date in all_dates]

                # 첫 번째 키워드의 ratio로 정규화
                first_keyword_data = api_data['results'][0]['data']
                for i, date_entry in enumerate(first_keyword_data):
                    first_keyword_ratio = max(date_entry['ratio'], 1e-6)  # ZeroDivisionError 방지
                    for result in api_data['results']:
                        result['data'][i]['ratio'] /= first_keyword_ratio
                return api_data
            else:
                raise Exception(f"API 요청 실패, 상태 코드: {rescode}")

        except urllib.error.HTTPError as e:
            raise Exception(f"HTTPError 발생: {e.code} - {e.reason} keywords: {keywords}")

        except urllib.error.URLError as e:
            raise Exception(f"URLError 발생: {e.reason}")

    def get_long_api_data(keywords, gender, ages):

        # 키워드 인자가 5개 초과 시 첫번째 키워드가 반복적으로 들어가도록 서브쿼리 설정
        sub_queries = [[keywords[0]] + keywords[i * 4 + 1:i * 4 + 5] for i in range((len(keywords) - 1) // 4 + 1)]
        with ThreadPoolExecutor() as executor:  # 병렬호출
            results = list(executor.map(lambda kw: SearchTrendData.get_api_data(kw, gender, ages), sub_queries))

        merged_results = {"results": []}
        first_keyword_done = False
        for result in results:
            for entry in result['results']:
                if entry['title'] == keywords[0]:
                    if not first_keyword_done:
                        merged_results['results'].append(entry)
                        first_keyword_done = True
                else:
                    merged_results['results'].append(entry)
        results = merged_results['results']

        date_ratios = {}
        for result in results:
            for entry in result['data']:
                date = entry['period']
                ratio = entry['ratio']
                if date not in date_ratios:
                    date_ratios[date] = []
                date_ratios[date].append(ratio)

        # 날짜별 데이터 수집
        datewise_data = {}
        for result in results:
            for entry in result['data']:
                date = entry['period']
                if date not in datewise_data:
                    datewise_data[date] = []
                datewise_data[date].append({"title": result["title"], "ratio": entry["ratio"]})
        results = merged_results['results']

        # 날짜별 정렬 및 순위 추가
        for date, entries in datewise_data.items():
            # ratio 기준 내림차순 정렬
            entries.sort(key=lambda x: x["ratio"], reverse=True)
            # 순위 부여
            for rank, entry in enumerate(entries, start=1):
                entry["rank"] = rank
        # 원래 데이터에 순위 추가
        for result in results:
            for entry in result['data']:
                date = entry['period']
                for ranked_entry in datewise_data[date]:
                    if ranked_entry["title"] == result["title"] and ranked_entry["ratio"] == entry["ratio"]:
                        entry["rank"] = ranked_entry["rank"]
                        break
        rank_data = {}
        for result in results:
            ranksum = 0
            for entry in result['data']:
                ranksum += entry['rank']
            rank_data[result['title']] = ranksum

        # 랭크 합산을 기준으로 순위 매기기
        sorted_rank_data = sorted(rank_data.items(), key=lambda x: x[1])

        # [{ keyword : string, rank : int }, ...] 형태의 배열 반환
        final_ranking = [{"keyword": kw, "rank": idx + 1} for idx, (kw, _) in enumerate(sorted_rank_data)]

        return final_ranking

    def get_final_df(keywords):
        result = pd.DataFrame()
        result['keyword'] = keywords
        result.set_index('keyword', inplace=True)
        age_g = ['10gen', '20gen', '30gen', '40gen', '50gen', '60gen']
        gen = ['m', 'f']
        ags = [['1', '2'], ['3', '4'], ['5', '6'], ['7', '8'], ['9', '10'], ['11']]

        # 병렬 처리용 함수 선언 (밖에 뽑지 마세요!)
        def process_group(g, j, age, age_g, keywords):
            temp = SearchTrendData.get_long_api_data(keywords, g, age) or []  # 빈 리스트 처리
            if g == 'm':
                gn = 'male'
            else:
                gn = 'female'
            col_name = f'{age_g[j]}_{gn}'
            results = []
            for k in temp:
                # rank가 None인 경우 기본값 0으로 대체
                rank = k.get('rank', 0) if k.get('rank') is not None else 0
                results.append((k['keyword'], col_name, rank))
            return results

        # 사전 초기화: 예상 컬럼 추가
        for g in gen:
            for j in range(len(ags)):
                col_name = f"{age_g[j]}_{'male' if g == 'm' else 'female'}"
                result[col_name] = pd.NA  # 기본 결측값 추가

        # 병렬 처리 수행
        with ThreadPoolExecutor() as executor:
            futures = []
            for g in gen:
                for j, age in enumerate(ags):
                    futures.append(executor.submit(process_group, g, j, age, age_g, keywords))
            for future in futures:
                processed_results = future.result()
                for keyword, col_name, rank in processed_results:
                    # keyword가 없으면 row 생성
                    if keyword not in result.index:
                        result.loc[keyword] = pd.NA
                    # rank가 None인 경우 기본값 0으로 대체
                    result.at[keyword, col_name] = rank if rank is not None else 0

        # 모든 결측값을 일관되게 처리 (NaN -> 제일 안좋은 값으로 변환)
        result.fillna(value=len(result), inplace=True)
        return result