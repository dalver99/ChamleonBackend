import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime

# BASE_DIR 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리 경로

# 경로 설정을 위한 헬퍼 함수
def get_path(*subdirs):
    """BASE_DIR 기준으로 하위 경로를 생성"""
    return os.path.join(BASE_DIR, *subdirs)

def load_file(filepath):
    with open(filepath, 'r') as file:
        return file.readline().strip()

# 1. API 요청 정보 설정
API_KEY = load_file(get_path("api_key.txt"))
BASE_URL = "http://openapi.seoul.go.kr:8088"
SERVICE = "citydata"
START_INDEX = 1
END_INDEX = 5
AREA_NM = "군자역"  # 요청할 지역 이름

# URL 생성
url = f"{BASE_URL}/{API_KEY}/xml/{SERVICE}/{START_INDEX}/{END_INDEX}/{AREA_NM}"

# 2. API 요청
response = requests.get(url)

# 2. 응답 확인
if response.status_code == 200:
    print("API 호출 성공!")
    xml_data = response.text
else:
    print(f"API 호출 실패! 상태 코드: {response.status_code}")
    xml_data = None

# 3. XML 데이터를 DataFrame으로 변환
if xml_data:
    root = ET.fromstring(xml_data)

    # 실시간 데이터를 위한 리스트
    live_data = []
    forecast_data = []
    weather_data = []
    forecast24_data = []
    
    for citydata in root.findall(".//CITYDATA"):
        # 실시간 인구 데이터 추출
        area_name = citydata.find("AREA_NM").text
        area_code = citydata.find("AREA_CD").text
        congestion_level = citydata.find(".//LIVE_PPLTN_STTS/AREA_CONGEST_LVL").text
        congestion_message = citydata.find(".//LIVE_PPLTN_STTS/AREA_CONGEST_MSG").text
        population_min = citydata.find(".//LIVE_PPLTN_STTS/AREA_PPLTN_MIN").text
        population_max = citydata.find(".//LIVE_PPLTN_STTS/AREA_PPLTN_MAX").text
        population_time = citydata.find(".//LIVE_PPLTN_STTS//PPLTN_TIME").text
        male_rate = citydata.find(".//LIVE_PPLTN_STTS//MALE_PPLTN_RATE").text
        female_rate = citydata.find(".//LIVE_PPLTN_STTS//FEMALE_PPLTN_RATE").text
        rate_0 = citydata.find(".//LIVE_PPLTN_STTS//PPLTN_RATE_0").text
        rate_10 = citydata.find(".//LIVE_PPLTN_STTS//PPLTN_RATE_10").text
        rate_20 = citydata.find(".//LIVE_PPLTN_STTS//PPLTN_RATE_20").text
        rate_30 = citydata.find(".//LIVE_PPLTN_STTS//PPLTN_RATE_30").text
        rate_40 = citydata.find(".//LIVE_PPLTN_STTS//PPLTN_RATE_40").text
        rate_50 = citydata.find(".//LIVE_PPLTN_STTS//PPLTN_RATE_50").text
        rate_60 = citydata.find(".//LIVE_PPLTN_STTS//PPLTN_RATE_60").text
        rate_70 = citydata.find(".//LIVE_PPLTN_STTS//PPLTN_RATE_70").text
        rate_resident = citydata.find(".//LIVE_PPLTN_STTS//RESNT_PPLTN_RATE").text
        rate_non_resident = citydata.find(".//LIVE_PPLTN_STTS//NON_RESNT_PPLTN_RATE").text
        replace_yn = citydata.find(".//LIVE_PPLTN_STTS//REPLACE_YN").text
        update_time = citydata.find(".//LIVE_PPLTN_STTS//PPLTN_TIME").text

        # 실시간 데이터를 리스트에 추가
        live_data.append({
            "지역 이름": area_name,
            "지역 코드": area_code,
            "혼잡도 수준": congestion_level,
            "혼잡도 메시지": congestion_message,
            "최소 인구": population_min,
            "최대 인구": population_max,
            "인구 시간": population_time,
            "남성 비율": male_rate,
            "여성 비율": female_rate,
            "10세 미만 비율": rate_0,
            "10대 비율": rate_10,
            "20대 비율": rate_20,
            "30대 비율": rate_30,
            "40대 비율": rate_40,
            "50대 비율": rate_50,
            "60대 비율": rate_60,
            "70대 비율": rate_70,
            "상주 인구 비율": rate_resident,
            "비상주 인구 비율": rate_non_resident,
            "대체 대에터 여부": replace_yn,
            "인구 데이터 업데이트 시간": update_time,
        })

    # 혼잡도 및 예측 인구 데이터 추출
    # 최상위 FCST_PPLTN 요소 찾기
    fcst_elements = citydata.findall(".//LIVE_PPLTN_STTS/FCST_PPLTN")

    for parent_fcst in fcst_elements:
        # 부모 FCST_PPLTN 내의 자식 FCST_PPLTN 요소 순회
        child_fcsts = parent_fcst.findall(".//FCST_PPLTN")
        for child_fcst in child_fcsts:
            fcst_time = child_fcst.find("FCST_TIME").text
            congestion_level = child_fcst.find("FCST_CONGEST_LVL").text
            ppltn_min = child_fcst.find("FCST_PPLTN_MIN").text
            ppltn_max = child_fcst.find("FCST_PPLTN_MAX").text
            
            # print(f"Time: {fcst_time}, Congestion: {congestion_level}, "
            #     f"Min Population: {ppltn_min}, Max Population: {ppltn_max}")
            
            forecast_data.append({
            "지역 이름": area_name,
            "지역 코드": area_code,
            "시간": fcst_time,
            "혼잡도 지수": congestion_level,
            "예측 최소 인구": int(ppltn_min),
            "예측 최대 인구": int(ppltn_max)
            })
            
    # 날씨 데이터 추출        
    for citydata in root.findall(".//CITYDATA"):
        # 기본 정보
        area_name = citydata.find("AREA_NM").text
        area_code = citydata.find("AREA_CD").text
        temp = citydata.find(".//WEATHER_STTS/TEMP").text
        sensible_temp = citydata.find(".//WEATHER_STTS/SENSIBLE_TEMP").text
        max_temp = citydata.find(".//WEATHER_STTS/MAX_TEMP").text
        min_temp = citydata.find(".//WEATHER_STTS/MIN_TEMP").text
        humidity = citydata.find(".//WEATHER_STTS/HUMIDITY").text
        wind_dirct = citydata.find(".//WEATHER_STTS/WIND_DIRCT").text
        wind_spd = citydata.find(".//WEATHER_STTS/WIND_SPD").text

        # 기상 관련 메시지 및 기타 정보
        pcp_msg = citydata.find(".//WEATHER_STTS/PCP_MSG").text
        sunrise = citydata.find(".//WEATHER_STTS/SUNRISE").text
        sunset = citydata.find(".//WEATHER_STTS/SUNSET").text
        uv_index_lvl = citydata.find(".//WEATHER_STTS/UV_INDEX_LVL").text
        uv_index = citydata.find(".//WEATHER_STTS/UV_INDEX").text
        uv_msg = citydata.find(".//WEATHER_STTS/UV_MSG").text
        pm25_index = citydata.find(".//WEATHER_STTS/PM25_INDEX").text
        pm25 = citydata.find(".//WEATHER_STTS/PM25").text
        pm10_index = citydata.find(".//WEATHER_STTS/PM10_INDEX").text
        pm10 = citydata.find(".//WEATHER_STTS/PM10").text
        air_idx = citydata.find(".//WEATHER_STTS/AIR_IDX").text
        air_idx_main = citydata.find(".//WEATHER_STTS/AIR_IDX_MAIN").text
        air_msg = citydata.find(".//WEATHER_STTS/AIR_MSG").text
        weather_time = citydata.find(".//WEATHER_STTS/WEATHER_TIME").text

        # 데이터 딕셔너리 추가
        weather_data.append({
            "지역 이름": area_name,
            "지역 코드": area_code,
            "기온": temp,
            "체감온도": sensible_temp,
            "최고온도": max_temp,
            "최저온도": min_temp,
            "습도": humidity,
            "풍향": wind_dirct,
            "풍속": wind_spd,
            "강수 관련 메세지": pcp_msg,
            "일출 시각": sunrise,
            "일몰 시각": sunset,
            "자외선 지수 단계": uv_index_lvl,
            "자외선 지수": uv_index,
            "자외선 메세지": uv_msg,
            "초미세먼지 지표": pm25_index,
            "초미세먼지 농도": pm25,
            "미세먼지 지표": pm10_index,
            "미세먼지 농도": pm10,
            "통합 대기 환경 등급": air_idx,
            "통합 대기 환경 지수": air_idx_main,
            "통합 대기 환경 등급별 메세지": air_msg,
            "날씨 데이터 업데이트 시간": weather_time
        })
        
    fcst24_elements = citydata.findall(".//WEATHER_STTS/FCST24HOURS")

    for parent_fcst in fcst24_elements:
        # 부모 FCST24HOURS 내의 자식 FCST24HOURS 요소 순회
        child_fcsts = parent_fcst.findall(".//FCST24HOURS")
        for child_fcst in child_fcsts:
            fcst_dt = child_fcst.find("FCST_DT").text
            fcst_datetime = datetime.strptime(fcst_dt, "%Y%m%d%H%M")
            formatted_fcst_datetime = fcst_datetime.strftime("%Y-%m-%d %H:%M")
            temp = child_fcst.find("TEMP").text
            precipitation = child_fcst.find("PRECIPITATION").text
            precip_type = child_fcst.find("PRECPT_TYPE").text
            rain_chance = child_fcst.find("RAIN_CHANCE").text
            sky_stts = child_fcst.find("SKY_STTS").text
            
            # print(f"Time: {fcst_time}, Congestion: {congestion_level}, "
            #     f"Min Population: {ppltn_min}, Max Population: {ppltn_max}")
            
            forecast24_data.append({
            "지역 이름": area_name,
            "지역 코드": area_code,
            "예보시간": formatted_fcst_datetime,
            "기온": temp,
            "강수량": precipitation,
            "강수형태": precip_type,
            "강수확률": rain_chance,
            "하늘상태": sky_stts
        })
    
    
    # Pandas DataFrame 생성
    live_df = pd.DataFrame(live_data)
    forecast_df = pd.DataFrame(forecast_data)
    weather_df = pd.DataFrame(weather_data)
    forecast24_df = pd.DataFrame(forecast24_data)
    
    # 4. CSV 파일로 저장
    live_df.to_csv("live_population.csv", index=False, encoding="utf-8-sig")
    forecast_df.to_csv("forecast_population.csv", index=False, encoding="utf-8-sig")
    weather_df.to_csv("weather_forecast.csv", index=False, encoding="utf-8-sig")
    forecast24_df.to_csv("weather_forecast24.csv", index=False, encoding="utf-8-sig")
    print("CSV 파일 저장 완료")
    
    
    
    
    # xml 구조 파악 코드
    # fcst_list = citydata.findall(".//WEATHER_STTS/FCST24HOURS")
    # for i, fcst in enumerate(fcst_list, start=1):
    #     print(f"Element {i}:")
    #     print(ET.tostring(fcst, encoding='unicode'))
    #     print("-" * 40)