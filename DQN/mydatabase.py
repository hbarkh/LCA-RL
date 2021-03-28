from sqlalchemy import create_engine

engine = create_engine('sqlite:///HyperParamStudies.db')

connection = engine.connect()
