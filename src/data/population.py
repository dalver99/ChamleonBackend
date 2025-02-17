from datetime import datetime, timedelta
from typing import List
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session
from src.model.population import PopulationStation

# 모든 인구 데이터
def get_all_populations(db: Session) -> list[PopulationStation]:
    db_populations = db.query(PopulationStation).all()
    return db_populations

# 특정 지역의 인구 데이터 생성
def create_population(population: PopulationStation, db: Session):
    population_data = PopulationStation(**population.model_dump())
    db.add(population_data)
    db.commit()
    db.refresh(population_data)
    return population_data


# 지역에 따른 인구 데이터 목록
def get_region(db: Session, region_id: int):
    query = db.query(PopulationStation).filter(PopulationStation.region_id == region_id)
    df = pd.read_sql(query.statement, db.bind)
    return df

# 지역, 시간에 따른 인구 데이터 목록
def get_region_and_time_range(
    db: Session, region_id: int, start_time: str, end_time: str):
    query = db.query(PopulationStation).filter(
        PopulationStation.region_id == region_id, 
        func.time(PopulationStation.datetime).between(start_time, end_time)
    )
    df = pd.read_sql(query.statement, db.bind)
    return df


# 성별 인구 데이터
def get_gender_population_data(
    db: Session, region_id: int, start_time: str, end_time: str):
    query = db.query(PopulationStation).filter(
        PopulationStation.region_id == region_id, 
        func.time(PopulationStation.datetime).between(start_time, end_time)
    )
    df = pd.read_sql(query.statement, db.bind)
    df["male_min_population"] = df["male_rate"] * df["min_population"] / 100
    df["male_max_population"] = df["male_rate"] * df["max_population"] / 100
    df["female_min_population"] = df["female_rate"] * df["min_population"] / 100
    df["female_max_population"] = df["female_rate"] * df["max_population"] / 100
    gender_df = df[[
        "datetime", "region_id", "male_min_population", "male_max_population",
        "female_min_population", "female_max_population"
    ]]
    return gender_df

# 나이대별 최소 인구 데이터프레임
def get_age_group_min_population_data(db: Session, region_id: int):
    query = db.query(PopulationStation).filter(
        PopulationStation.region_id == region_id
    )
    df = pd.read_sql(query.statement, db.bind)
    for age_group in ["10_gen", "20_gen", "30_gen", "40_gen", "50_gen", "60_gen", "70_gen"]:
        df[age_group] = df[age_group] * df["min_population"] / 100
    rename_columns = {
        "10_gen": "gen_10",
        "20_gen": "gen_20",
        "30_gen": "gen_30",
        "40_gen": "gen_40",
        "50_gen": "gen_50",
        "60_gen": "gen_60",
        "70_gen": "gen_70",
    }
    df.rename(columns=rename_columns, inplace=True)
    age_group_columns = [
        "datetime", "region_id"
    ] + list(rename_columns.values())
    age_group_df = df[age_group_columns]
    return age_group_df

# 나이대별 최대 인구 데이터프레임
def get_age_group_max_population_data(db: Session, region_id: int):
    query = db.query(PopulationStation).filter(
        PopulationStation.region_id == region_id
    )
    df = pd.read_sql(query.statement, db.bind)
    for age_group in ["10_gen", "20_gen", "30_gen", "40_gen", "50_gen", "60_gen", "70_gen"]:
        df[age_group] = df[age_group] * df["max_population"] / 100
    rename_columns = {
        "10_gen": "gen_10",
        "20_gen": "gen_20",
        "30_gen": "gen_30",
        "40_gen": "gen_40",
        "50_gen": "gen_50",
        "60_gen": "gen_60",
        "70_gen": "gen_70",
    }
    df.rename(columns=rename_columns, inplace=True)
    age_group_columns = [
        "datetime", "region_id"
    ] + list(rename_columns.values())
    age_group_df = df[age_group_columns]
    return age_group_df