import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import blogabet.configuration as configuration
import blogabet.database as database
import blogabet.model as model


def make_engine(url):
    return create_engine(url, echo=True)


def make_session(engine):
    session = sessionmaker(bind=engine)
    return sqlalchemy.orm.Session()


def bind_model(engine, declarative_base):
    declarative_base.metadata.bind = engine
    declarative_base.metadata.create_all()


def update(Session, item):
    database.save2db(Session, item)


def prepare_database(url, declarative_base):
    engine = make_engine(url)
    Session = make_session(engine)
    bind_model(url, declarative_base)
    return Session


if __name__ == '__main__':
    Session = prepare_database(configuration.database_url, model.Base)
