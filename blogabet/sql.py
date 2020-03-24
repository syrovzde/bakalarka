import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import blogabet.configuration as configuration
import blogabet.database as database
import blogabet.model as model

"""
For easier work with database
"""
def make_engine(url):
    """creates engine linking to database"""
    return create_engine(url, echo=True)


def make_session(engine):
    """makes session from engine"""
    session = sessionmaker(bind=engine)
    return sqlalchemy.orm.Session()


def bind_model(engine, declarative_base):
    """bind engine to declarative model and creates metadata"""
    declarative_base.metadata.bind = engine
    declarative_base.metadata.create_all()


def update(Session, item):
    """ updates item in database"""
    save2db(Session, item)


def prepare_database(url, declarative_base):
    engine = make_engine(url)
    Session = make_session(engine)
    bind_model(url, declarative_base)
    return Session


def save2db(session, item):
    """"Save the given item into DB through the open session"""
    try:
        # MERGE will perform UPDATE instead of plain add (will overwrite old data with new instead of failing)
        # thus it will update the data with the latest version, which is mostly what we want
        session.merge(item)
        session.commit()
    except:
        session.rollback()
        raise


if __name__ == '__main__':
    Session = prepare_database(configuration.database_url, model.Base)
