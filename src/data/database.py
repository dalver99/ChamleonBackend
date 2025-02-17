from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from starlette.config import Config
from dotenv import load_dotenv
import os

load_dotenv('.env')
SQLALCHEMY_DATABASE_URL = os.getenv('SQLALCHEMY_DATABASE_URL')

if SQLALCHEMY_DATABASE_URL.startswith('sqlite'):
    engine = create_engine(SQLALCHEMY_DATABASE_URL,
                           connect_args={'check_same_thread': False})
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL,
                           connect_args={
                            "ssl_ca": "./DigiCertGlobalRootCA.crt.pem"
    })

engine = create_engine("sqlite:///./test.db", connect_args={'check_same_thread': False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

nameing_convention = {
    "ix" : 'ix_%(column_0_label)s',
    "uq" : "uq_%(table_name)s_%(column_0_name)s",
    "ck" : "ck_%(table_name)s_%(column_0_name)s",
    "fk" : "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk" : "pk_%(table_name)s"
}
Base.metadata = MetaData(naming_convention=nameing_convention)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()