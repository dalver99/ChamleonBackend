![Image](https://github.com/user-attachments/assets/6befe2a4-3b15-4b82-ac42-e478bca600d5)
# ChamelNeon Backend
- 본 문서는 KT Aivle 6기 3반 9조의 Big Project인 AI 기반 광고 매니지먼트 플랫폼 ChamelNeon의 기능적인 부분을 소개합니다.
- 현재 버전은 광고 순서리스트 제공 서비스와 광고 타겟 추천 서비스만 지원합니다. 

![Image](https://github.com/user-attachments/assets/8bc9c27d-97d0-41bd-8118-eadfcec99a7e)

# Demo Video
![Image](https://github.com/user-attachments/assets/9f413623-c2af-4b6c-ba75-e6815a19fc05)

  
# Install
- 해당 문서에 포함된 코드를 git clone 등으로 다운받은 후, 프롬프트에서 아래 명령어를 통해 필요한 파이썬 라이브러리를 설치합니다.
```
pip install -r requirements.txt
```
# 데이터베이스 연결 설정
- 다음과 같이 json 파일을 생성해주세요.
`process/SQL_parameter.json`으로 존재해야 하며, 아래와 같은 형식을 따릅니다.   
해당 파일은 `AdvExposureGenerator`, `AdvTargetRecommander`, `gazevolo_fastapi`에서 사용됩니다.

  ```
  {
  "host":"Your host (e.g., big09db.mysql.database.azure.com)",
  "port": Your Port number (e.g., 3306 for MySQL),
  "username":"Your username (e.g., big09_test)",
  "password":"Your password (e.g., testpassword)",
  "database":"Your database (e.g., big092)"
  }
  ```

# Usage: AdvExposureGenerator
- `process/AdvExposureGenerator/RecAlg_demo_prototype.ipynb`에서 자세한 구조를 확인할 수 있습니다.      

![Image](https://github.com/user-attachments/assets/a840f278-7b87-4bfa-a765-ac9d78238603)

    
### naver api key 설정
- 위 데모파일을 정상적으로 실행시키기 위해서는 naver api key가 json 파일로 동일경로에 존재해야합니다.
- naver api key json 파일은 `process/AdvExposureGenerator/naver_api_key.json`으로 존재해야하며,    
아래와 같은 형식을 따릅니다.   
  ```
  {
  "url":"https://openapi.naver.com/v1/datalab/search",
  "client_id":"Your Naver Client ID",
  "client_secret":"Your Naver Client Secret"
  }
  ```   
### 프롬프트 실행 방법
프롬프트에서 실행시키기 위해서는 아래 명령어를 사용하세요:
  ```
  cd process/AdvExposureGenerator
  python exposure_main.py --loc location number
  ```
### Arguments 설명
| Argument Name   | Type    | Default Value  | Required | Description                                                                 |
|-----------------|---------|----------------|----------|-----------------------------------------------------------------------------|
| `--mode`        | `bool`  | `True`         | No       | LSTM을 통해 예측된 인구 데이터를 사용할 경우 `True`로 설정합니다.            |
| `--loc`         | `int`   | None           | Yes      | 데이터베이스에 저장된 위치 번호를 입력합니다. 이 값은 필수입니다.             |
| `--start_date`  | `str`   | `'2025-01-08'` | No       | 시작 날짜를 입력합니다. 날짜 형식은 `YYYY-MM-DD`로 입력해야 합니다.           |
| `--search_option`| `bool`  | `False`        | No       | 네이버 검색 트렌드 데이터를 사용할 경우 `True`로 설정합니다.                 |
| `--DB_upload`   | `bool`  | `False`        | No       | 생성된 노출 리스트를 메인 데이터베이스에 업로드하려면 `True`로 설정합니다.     |


<br>

# Usage: AdvTargetRecommander
- `process/AdvTargetRecommander/Recommand target using PCAFA demo.ipynb`에서 자세한 구조를 확인할 수 있습니다.     

![Image](https://github.com/user-attachments/assets/5a529679-3456-4848-adf1-d05e9c483a51)


### 프롬프트 실행 방법
프롬프트에서 실행시키기 위해서는 아래 명령어를 사용하세요:
  ```
  python process/AdvTargetRecommander/main_pcafa.py --target Company Name --threshold float
  ```

### Arguments 설명
| Argument Name   | Type    | Default Value  | Required | Description                                                                 |
|-----------------|---------|----------------|----------|-----------------------------------------------------------------------------|
| `--target`        | `str`  | `None`         | Yes       | 서비스에 등록된 `title`을 입력합니다. 필수 값입니다.            |
| `--threshold`         | `float`   | `0.8`           | No      | 	임계값(Threshold)을 설정합니다. 0.8 이하의 값을 권장하며, 기본값은 `0.8`입니다.             |


<br>

# Usage: gazevolo_fastapi
- `process/gazevolo_fastapi/`폴더내 에서 자세한 구조를 확인할 수 있습니다. 

### 1. IMGUR API 연결 설정
  - 다음과 같이 json 파일을 생성해주세요.
  `process/gazevolo_fastapi/imgur_api.json`으로 존재해야 하며, 아래와 같은 형식을 따릅니다.   
       
  [IMGUR](https://imgur.com/)에서 개인키를 발급 후 사용가능합니다. 아래는 예시입니다. 

```
{
    "client_id": "Your client ID (e.g., 76b478ad23c26)"
}
```

### 2. Mivolo 모델 다운로드
- ['https://github.com/WildChlamydia/MiVOLO'](https://github.com/WildChlamydia/MiVOLO) 에서 `yolov8x_person_face.pt` 과 `mivolo_imbd.pth.tar`를 다운로드해주세요.
- 해당 파일을 gazevolo_fastapi 폴더 내 models 폴더를 생성 후 그 안에 넣어주세요.
- 아래 사진은 예시입니다.

  ![Image](https://github.com/user-attachments/assets/01c87e45-f072-4ab5-a94b-35b7f6d34302)

### 3. 실행
- 위 사항들이 모두 완료가 되었다면 commend창에서 `uvicorn main:app --reload`로 `http://127.0.0.1:8000/docs`에 접근해 실행해보실 수 있습니다. 

<br>

# Related Works
- 데이터 분석 및 자체 모델 훈련: https://github.com/UnoesmiK/KT6_BP9_DataAnalysis.git
- 프론트 엔드: https://github.com/dalver99/team9_bigproject.git
- 모델 : https://github.com/dalver99/gazevolo.git

# ChamelNeon - Data Analysis
![Image](https://github.com/user-attachments/assets/6befe2a4-3b15-4b82-ac42-e478bca600d5)
## 프로젝트 개요

이 프로젝트의 데이터 분석은 KT Aivle 6기 3반 9조의 Big Project인 AI 기반 광고 매니지먼트 플랫폼 **ChamelNeon**의 효율적인 광고 타겟 추천 및 광고 성과 분석을 위한 기반을 마련하는 데 중점을 두었습니다. 

데이터 분석 과정은 **데이터 전처리**, **데이터 분석**, **딥러닝 모델**을 활용하여 광고 타겟팅 최적화 및 광고 효과 향상에 기여하는 것을 목표로 했습니다.

## 프로젝트 목적
- 유동인구, 기상 상황, 상권 분포 데이터를 활용하여 광고 최적화
- 광고 타겟 그룹 설정 및 맞춤형 광고 송출
- AI 기반의 데이터 분석을 통해 광고 효율 극대화

## 프로젝트 구조
```plaintext
KT6_BP9_DATAANALYSIS/
│── Data_Analysis/        # 데이터 분석 관련 코드
│── Data_Preprocessing/   # 데이터 전처리 관련 코드 
│── DL_models/            # 딥러닝 모델 관련 코드
│── README.md             # 프로젝트 개요 및 설명
```

## 🛠 데이터 전처리
데이터 분석과 모델 학습을 위해 원본 데이터를 정리하고 변환하는 과정입니다.
- **목적:** 원본 데이터를 가공하여 분석 및 모델 학습이 가능한 형식으로 변환
- **작업 내용:** 특정 지하철역의 생활인구수 데이터를 추출하고, 날짜별로 집계하여 csv파일로 저장하는 작업을 수행합니다.
- **관련 파일:** LPD_target_generator_all.py, LPD_target_generator.py, LPD_target.ipynb 등
- **주요 전처리 과정:**
1. 데이터 병합 및 통계 분석:
  여러 유동 인구 데이터 파일을 병합하고, 각 지역에 대한 통계 분석을 수행하여 데이터의 특성을 파악합니다.
2. 시계열 분석:
  특정 기간에 따른 유동 인구 변화나 패턴을 분석하고, 광고 타겟 선정에 도움이 될 수 있는 시간대별/날짜별 분석을 진행합니다.
3. 지역별 유동 인구 특성 분석:
  특정 지역의 유동 인구를 분석하고, 광고 타겟에 유용한 인사이트를 도출합니다. 이 과정은 광고 노출 효과를 극대화하는 데 활용될 수 있습니다.
  
## 📊 데이터 분석
AI를 활용한 광고 추천 및 효과 예측 모델을 개발하는 과정입니다.
- **목적:** 유동인구, 기상, 상권 데이터를 분석하여 광고 타겟팅 및 성과 예측을 위한 인사이트 도출
- **작업 내용:** 지하철역, 기상, 상권별 유동 인구 데이터를 분석하고, 각 지역의 유동 인구 특성과 패턴을 기상 조건 및 상권에 맞춰 분석하는 작업을 수행합니다.
- **관련 파일:** #1 유동 인구 분석_250211.ipynb, #2 기상-유동 인구 분석.ipynb, #3 상권-유동 인구 분석_250211.ipynb, gazed_data.ipynb 등
- **분석 과정:**
1. 유동인구 분석:
   특정 지하철역의 시간대별 유동인구 패턴을 분석합니다.
2. 기상-유동인구 분석:
   기상조건과 유동인구의 상관관계가 없으므로 유동인구 데이터만 활용한 유동인구 예측 모델을 구축합니다.
3. 상관-유동인구 분석:
   특정 상권 내 유동인구 특성을 분석하여 지역별로 독립된 유동인구 예측 모델을 구축합니다.

## 🤖 딥러닝 모델
AI를 활용한 유동 인구 예측 모델을 개발하는 과정입니다.
- **목적:** LSTM 기반의 유동인구 예측모델을 제작하여 24시간 이후의 유동인구를 성별/연령별로 예측하는 모델 개발
- **작업 내용:** LSTM 모델을 활용하여 유동 인구 예측을 학습하고, 평가하며, 성과 지표(RMSE, MAE, R²)를 계산하여 최적화했습니다.
- **관련 파일:** LPD_lstm.py 등
- **모델 구성:**
1. LSTM을 사용해 과거 24시간 데이터를 기반으로 유동 인구 예측을 진행
2. 예측값을 MinMaxScaler로 역변환하여 실제 값과 비교 후 성능 평가
3. RMSE, MAE, R² 지표를 사용하여 모델 성능을 평가하고, 예측 정확도를 측정
4. 파인튜닝을 통해 모델 성능을 최적화
