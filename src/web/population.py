from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette import status
from src.model.population import PopulationStation
from src.service.population import get_region, get_region_and_time_range
from src.data.database import get_db
from src.schema.population import AgeGroupPopulationResponse, GenderPopulationResponse, PopulationRequest, PopulationResponse
from src.service import population as service

router = APIRouter(prefix = "/populations")

@router.get("/list", response_model=list[PopulationResponse])
def get_all_populations(db: Session = Depends(get_db)):
    """
    모든 region_id의의 인구 데이터를 반환
    """
    db_populations = service.get_region(db)
    return db_populations

@router.post("/create_population", response_model=PopulationResponse)
def create_population(population: PopulationRequest, db: Session = Depends(get_db)):
    """
    특정 region_id의 인구 데이터를 생성
    """
    population_data = service.create_population(population, db)
    return population_data

@router.get("/region/{region_id}", response_model=list[PopulationResponse])
def get_population_by_region(region_id: int, db: Session = Depends(get_db)):
    """
    특정 region_id의 데이터를 반환
    """
    df = get_region(db, region_id)
    return df.to_dict(orient="records")

@router.get("/get_region_id_time", response_model=list[PopulationResponse])
def get_population_by_region_and_time(
    region_id: int,
    start_time: str,
    end_time: str,
    db: Session = Depends(get_db)
):
    """
    특정 region_id와 시간 범위의 데이터를 반환
    """
    df = get_region_and_time_range(db, region_id, start_time, end_time)
    return df.to_dict(orient="records")

@router.get("/gender_population_data", response_model=List[GenderPopulationResponse])
def get_gender_population_data(
    region_id: int,
    start_time: str,
    end_time: str,
    db: Session = Depends(get_db)
):
    """
    특정 region_id와 시간 범위의 성별 데이터를 반환
    """
    df = service.get_gender_population_data(db, region_id, start_time, end_time)

    response_data = df.to_dict(orient="records")
    return response_data

@router.get("/age_min_population_data", response_model=List[AgeGroupPopulationResponse])
def get_age_group_min_population_data(
    region_id: int,
    db: Session = Depends(get_db)
):
    """
    특정 region_id의 연령대별 최소 인구 데이터를 반환
    """
    df = service.get_age_group_min_population_data(db, region_id)

    response_data = df.to_dict(orient="records")
    return response_data

@router.get("/age_max_population_data", response_model=List[AgeGroupPopulationResponse])
def get_age_group_max_population_data(
    region_id: int,
    db: Session = Depends(get_db)
):
    """
    특정 region_id의 연령대별 최대 인구 데이터를 반환
    """
    df = service.get_age_group_max_population_data(db, region_id)

    response_data = df.to_dict(orient="records")
    return response_data