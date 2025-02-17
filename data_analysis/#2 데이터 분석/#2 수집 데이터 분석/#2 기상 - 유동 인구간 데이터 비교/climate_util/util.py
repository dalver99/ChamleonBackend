import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates


def import_climate_data(stn, tm1, tm2, api_key):
    url = "https://apihub.kma.go.kr/api/typ01/url/awsh.php?"
    params = {
        "stn": stn,
        "help": "0",
        "tm1": tm1,
        "tm2": tm2,
        "authKey": api_key
    }
    
    response = requests.get(url, params=params)
    raw_data = response.text.split('\n')
    
    # print(f"API Response: {response.text[:500]}...")  # 응답 내용 확인
    
    if len(raw_data) < 3:
        print(f"Warning: Insufficient data received for station {stn} from {tm1} to {tm2}")
        return pd.DataFrame()  # 빈 데이터프레임 반환
    
    raw_data = raw_data[1:-2]
    if len(raw_data) > 0:
        raw_data[0] = raw_data[0][2:]
    if len(raw_data) > 1:
        raw_data[1] = raw_data[1][2:]
    raw_data = raw_data[2:]
    cols = ['Datetime', '지점번호', '기온', '풍향', '풍속', '일 강수량', '1시간 강수량', '습도', '기압', 'PS']
    data = []
    
    for row in raw_data:
        columns = row.split()
        if len(columns) == len(cols):
            data.append(columns)
    
    if not data:
        print(f"Warning: No valid data found for station {stn} from {tm1} to {tm2}")
        return pd.DataFrame()  # 빈 데이터프레임 반환
    
    df = pd.DataFrame(data, columns=cols)
    df = df[['Datetime', '지점번호', '기온', '풍속', '1시간 강수량', '습도']]
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y%m%d%H%M')
    df.set_index('Datetime', inplace=True)
    return df


def import_station_num(target):
    loc_df = pd.read_csv('./climate_util/Area to station final.csv', index_col=0)
    stn = loc_df.loc[loc_df['Area name']==target, 'Station Code'].values[0]
    return stn


def analyze_population_data(target):  # ex) 군자역
    years = range(2020, 2025)  # 2020부터 2024까지
    df_total = pd.DataFrame()

    for year in years:
        year_data = f'LPD{year}_target_final.csv'
        df_pop, stn = import_population_data(target, year_data)
        
        df_pop['datetime'] = pd.to_datetime(
            df_pop.index.astype(str) + 
            df_pop['시간대구분'].astype(str).str.zfill(2), 
            format='%Y%m%d%H'
        )
        df_pop.set_index('datetime', inplace=True)
        df_pop = df_pop.drop(columns=['시간대구분'])
        
        api_path = './climate_util/climate_api_key.txt'
        with open(api_path, 'r') as f:
            api_key = f.readline().strip()
        
        tm1 = f"{year}01010000"
        tm2 = f"{year}12312300"
        df_climate = import_climate_data(stn, tm1, tm2, api_key)
        
        df_year = df_pop.copy()
        df_year = df_year.join(df_climate, how='left')
        
        df_total = pd.concat([df_total, df_year])

    df_total.sort_index(inplace=True)
    
    # 시각화 함수들
    plot_yearly_gender_distribution(df_total, target)         # 연도별 평균 성별 분포
    plot_yearly_age_distribution(df_total, target)            # 연도별 평균 나이대 분포
    plot_correlation_heatmap(df_total, target)                # 상관관계 히트맵
    plot_daily_time_series(df_total, target)                  # 일별 시계열 분석
    plot_monthly_time_series(df_total, target)                # 월별 시계열 분석
    plot_demographic_population_pattern(df_total, target)     # 요일/시간대/나이 및 성별 별 인구 패턴
    
    
def plot_yearly_gender_distribution(df_total, target):
    # 성별 데이터 생성
    df_total['male'] = df_total['10gen_male'] + df_total['20gen_male'] + df_total['30gen_male'] + \
                    df_total['40gen_male'] + df_total['50gen_male'] + df_total['60gen_male']
    df_total['female'] = df_total['10gen_female'] + df_total['20gen_female'] + df_total['30gen_female'] + \
                        df_total['40gen_female'] + df_total['50gen_female'] + df_total['60gen_female']

    # 연도별 평균 성별 분포 계산
    gender_by_year = df_total.groupby(df_total.index.year)[['male', 'female']].mean()

    # 시각화
    plt.figure(figsize=(12, 6))
    gender_by_year.plot(kind='bar', stacked=True)
    plt.title('연도별 평균 성별 분포')
    plt.xlabel('연도')
    plt.ylabel('평균 인구')
    plt.legend(title='성별')
    plt.show()
    
    
def plot_yearly_age_distribution(df_total, target):
    age_columns = ['10gen_male', '10gen_female', '20gen_male', '20gen_female', 
                '30gen_male', '30gen_female', '40gen_male', '40gen_female', 
                '50gen_male', '50gen_female', '60gen_male', '60gen_female']

    age_by_year = df_total.groupby(df_total.index.year)[age_columns].mean()

    plt.figure(figsize=(12, 6))
    age_by_year.plot(kind='area', stacked=True)
    plt.title('연도별 평균 나이대 분포')
    plt.xlabel('연도')
    plt.ylabel('평균 인구')
    plt.legend(title='나이대 및 성별', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    
def plot_correlation_heatmap(df_total, target):
    age_columns = ['10gen_male', '10gen_female', '20gen_male', '20gen_female', 
                '30gen_male', '30gen_female', '40gen_male', '40gen_female', 
                '50gen_male', '50gen_female', '60gen_male', '60gen_female']
    
    correlation = df_total[['총생활인구수', 'male', 'female'] + age_columns + ['기온', '풍속', '1시간 강수량', '습도']].corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('변수 간 상관관계 히트맵')
    plt.tight_layout()
    plt.show()

    
def plot_daily_time_series(df_total, target):
    # 일별 데이터로 리샘플링
    daily_total = df_total['총생활인구수'].resample('D').mean()

    plt.figure(figsize=(14, 6))
    plt.plot(daily_total.index, daily_total)
    plt.title(f'{target} 일별 평균 총생활인구수 시계열 분석')
    plt.xlabel('날짜')
    plt.ylabel('평균 총생활인구수')
    
    # x축 레이블 설정
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%m'))
    
    plt.gcf().autofmt_xdate()  # 날짜 레이블 자동 포맷
    plt.grid(True)
    plt.show()
    
    # 일별 데이터 사용 -> 시계열 분해
    period = 7
    result = seasonal_decompose(daily_total, model='additive', period=period)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16))
    result.observed.plot(ax=ax1)
    ax1.set_title(f'{target} 관측된 데이터 (일별, 주기={period})')
    result.trend.plot(ax=ax2)
    ax2.set_title('추세')
    result.seasonal.plot(ax=ax3)
    ax3.set_title('계절성')
    result.resid.plot(ax=ax4)
    ax4.set_title('잔차')
    
    # 각 서브플롯의 x축 레이블 설정
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))
    
    plt.tight_layout()
    plt.gcf().autofmt_xdate()  # 날짜 레이블 자동 포맷
    plt.show()

    
def plot_monthly_time_series(df_total, target):
    # 월별 추세를 확인하기 위한 그래프
    monthly_total = df_total['총생활인구수'].resample('M').mean()

    plt.figure(figsize=(14, 6))
    plt.plot(monthly_total.index, monthly_total)
    plt.title(f'{target} 월별 평균 총생활인구수 시계열 분석')
    plt.xlabel('날짜')
    plt.ylabel('평균 총생활인구수')
    
    # x축 레이블 설정
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%m'))
    
    plt.gcf().autofmt_xdate()  # 날짜 레이블 자동 포맷
    plt.grid(True)
    plt.show()
    
    # 월별 데이터 사용 -> 시계열 분해
    period = 12
    result = seasonal_decompose(monthly_total, model='additive', period=period)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16))
    result.observed.plot(ax=ax1)
    ax1.set_title(f'{target} 관측된 데이터 (월별, 주기={period})')
    result.trend.plot(ax=ax2)
    ax2.set_title('추세')
    result.seasonal.plot(ax=ax3)
    ax3.set_title('계절성')
    result.resid.plot(ax=ax4)
    ax4.set_title('잔차')
    
    # 각 서브플롯의 x축 레이블 설정
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))
    
    plt.tight_layout()
    plt.gcf().autofmt_xdate()  # 날짜 레이블 자동 포맷
    plt.show()

    
def plot_demographic_population_pattern(df_total, target):
    # 요일과 시간 컬럼 추가
    df_total['요일'] = df_total.index.dayofweek  # 0~6 = 월~일
    df_total['시간'] = df_total.index.hour

    # 연령대와 성별 컬럼 리스트
    age_gender_cols = ['10gen_male', '10gen_female', '20gen_male', '20gen_female', 
                    '30gen_male', '30gen_female', '40gen_male', '40gen_female', 
                    '50gen_male', '50gen_female', '60gen_male', '60gen_female']

    # 요일 및 시간대별 연령/성별 평균 인구수 계산
    grouped_data = df_total.groupby(['요일', '시간'])[age_gender_cols].mean()

    # 데이터 재구성
    melted_data = grouped_data.reset_index().melt(id_vars=['요일', '시간'], 
                                                value_vars=age_gender_cols, 
                                                var_name='연령_성별', 
                                                value_name='평균인구수')

    # 연령대와 성별 분리
    melted_data[['연령대', '성별']] = melted_data['연령_성별'].str.extract(r'(\d+gen)_(\w+)')

    # 히트맵 그리기 함수
    def plot_heatmap(data, title, xlabel, ylabel):
        plt.figure(figsize=(15, 8))
        sns.heatmap(data, cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': '평균 인구수'})
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    # 1. 전체 연령/성별 요일 및 시간대별 평균 인구수
    overall_heatmap = melted_data.groupby(['요일', '시간'])['평균인구수'].sum().unstack()
    plot_heatmap(overall_heatmap, '요일 및 시간대별 평균 총생활인구수', '시간', '요일')

    # 2. 성별에 따른 요일 및 시간대별 평균 인구수
    for gender in ['male', 'female']:
        gender_data = melted_data[melted_data['성별'] == gender]
        gender_heatmap = gender_data.groupby(['요일', '시간'])['평균인구수'].sum().unstack()
        plot_heatmap(gender_heatmap, f'요일 및 시간대별 {gender} 평균 생활인구수', '시간', '요일')

    # 3. 연령대별 요일 및 시간대별 평균 인구수
    for age_group in ['10gen', '20gen', '30gen', '40gen', '50gen', '60gen']:
        age_data = melted_data[melted_data['연령대'] == age_group]
        age_heatmap = age_data.groupby(['요일', '시간'])['평균인구수'].sum().unstack()
        plot_heatmap(age_heatmap, f'요일 및 시간대별 {age_group} 평균 생활인구수', '시간', '요일')

    # 4. 연령대 및 성별 비교 (막대 그래프)
    plt.figure(figsize=(15, 8))
    sns.barplot(x='연령대', y='평균인구수', hue='성별', data=melted_data)
    plt.title('연령대 및 성별 평균 생활인구수')
    plt.xlabel('연령대')
    plt.ylabel('평균 인구수')
    plt.legend(title='성별')
    plt.tight_layout()
    plt.show()