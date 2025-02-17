from src.schema.population import AgeGroupPopulationResponse, GenderPopulationResponse
from src.model.population import PopulationStation
import src.data.population as data
from sqlalchemy.orm import Session


def create_population(db: Session, population: PopulationStation,):
    return data.create_population(db, population)

def get_region(db: Session, region_id: int) -> list[PopulationStation]:
    return data.get_region(db, region_id)

def get_region_and_time_range(db: Session, region_id: int, start_time: str, end_time: str) -> list[PopulationStation]:
    return data.get_region_and_time_range(db, region_id, start_time, end_time)

def get_gender_population_data(db: Session, region_id: int, start_time: str, end_time: str) -> list[GenderPopulationResponse]:
    return data.get_gender_population_data(db, region_id, start_time, end_time)

def get_age_group_min_population_data(db: Session, region_id: int) -> list[AgeGroupPopulationResponse]:
    return data.get_age_group_min_population_data(db, region_id)

def get_age_group_max_population_data(db: Session, region_id: int) -> list[AgeGroupPopulationResponse]:
    return data.get_age_group_max_population_data(db, region_id)