from configparser import ConfigParser
from sqlalchemy import create_engine
from .schema_engine import SchemaEngine


def schema_get() -> str:
    """
     construct M-Schema from database.
    """
    config = ConfigParser()
    config_file = './config/config.ini'
    config.read(config_file)
    db_user_name = config['database']['user']
    db_pwd = config['database']['password']
    db_host = config['database']['host']
    db_port = config['database']['port']
    db_name = config['database']['dbname']
    db_engine = create_engine(f"mysql+pymysql://{db_user_name}:{db_pwd}@{db_host}:{db_port}/{db_name}")
    schema_engine = SchemaEngine(engine=db_engine, db_name=db_name)
    mschema = schema_engine.mschema
    mschema_str = mschema.to_mschema()
    return mschema_str
