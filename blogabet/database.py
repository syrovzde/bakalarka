import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def db_connect(db_url=''):
    """
    Performs database connection using either given database url string or default database setting from settings.py.
    Returns sqlalchemy engine instance.
    """
    logging.info("Creating an SQLAlchemy engine at URL '{db_url}'".format(db_url=db_url))

    return create_engine(db_url)


def init_db(db_url=''):
    """
    Initializes database connection and sessionmaker.
    Creates all registered tables only IF they don't exist.
    """
    engine = db_connect(db_url)
    return sessionmaker(bind=engine), engine


