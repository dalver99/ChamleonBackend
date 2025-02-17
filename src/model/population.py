from sqlalchemy import Column, Integer, Float, String, DateTime
from src.data.database import Base

from sqlalchemy import Column, Integer, Float, String, DateTime

class PopulationStation(Base):
    __tablename__ = 'population'
    
    datetime = Column(DateTime(timezone=True), primary_key=True, nullable=False, index=True)  
    region_id = Column(Integer, primary_key=True, nullable=False)              
    male_rate = Column(Float, nullable=True)                 
    female_rate = Column(Float, nullable=True)               
    area_congest = Column(String(255), nullable=True)         
    gen_10 = Column('10_gen', Float, nullable=True)
    gen_20 = Column('20_gen', Float, nullable=True) 
    gen_30 = Column('30_gen', Float, nullable=True) 
    gen_40 = Column('40_gen', Float, nullable=True) 
    gen_50 = Column('50_gen', Float, nullable=True)  
    gen_60 = Column('60_gen', Float, nullable=True)  
    area_congest_pre_3 = Column(Float, nullable=True)         
    gen_70 = Column('70_gen', Float, nullable=True) 
    min_population = Column(Integer, nullable=True)           
    max_population = Column(Integer, nullable=True)           
          
