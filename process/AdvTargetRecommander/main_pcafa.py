import pandas as pd
from sqlalchemy import create_engine
from util_data import *
from util_pcafa import *
import argparse
import json

pd.set_option('future.no_silent_downcasting', True)

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    parser.add_argument("--target", type=str, default=None, required=True)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--loc_num", type=int, default=91)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    target = args.target
    threshold = args.threshold
    print(target, threshold)
    # MySQL 데이터베이스 연결
    with open(f'./SQL_parameter.json', 'r') as file:
        par = json.load(file)

    # MySQL 연결 정보 설정
    host = par['host']
    port = par['port']
    username = par['username']
    password = par['password']
    database = par['database']

    # 비밀번호 URL 인코딩
    encoded_password = urllib.parse.quote_plus(password)

    # SQLAlchemy 엔진 생성 및 필요한 데이터 가져오기
    engine = create_engine(f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database}")
    adv_gaze, adv_info, adv_target = import_data(engine)
    adv_id, adv_gt = edit_data(adv_info, adv_gaze, target)
    data_pre = generate_input(adv_gt)
    n_comp, copm_values, data_pre_ = apply_PCA(data_pre)
    print('70% 이상 성분을 지닌 주성분 수:', n_comp)
    # 응시하는 보행자 분석결과 기반로 재추천 리스트 생성
    rt_df,fa_df = Apply_FA(n_comp, threshold, data_pre_)
    raw_t, rec_t = export_targets(adv_target, adv_id, rt_df)
    fa_pairplot(fa_df, False)

    print(f"상품명: {target}")
    print('\n')
    print('설정된 광고대상')
    print(f'성별: {format_gender(raw_t)}')
    print(f'연령대: {", ".join(format_age_groups(raw_t))}')
    print('\n')
    print('추천 광고대상')
    print(f'성별: {format_gender(rec_t)}')
    print(f'연령대: {", ".join(format_age_groups(rec_t))}')

if __name__ == "__main__":
    main()