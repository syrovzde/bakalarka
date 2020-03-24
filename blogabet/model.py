from sqlalchemy import Column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import Text, Float, Integer

Base = declarative_base()

"""
model of simple database
"""

class Bettor(Base):
    __tablename__ = "bettor_html"
    name = Column(Text, primary_key=True)
    HTML = Column(Text, nullable=False)
    url = Column(Text)


class Matches(Base):
    __tablename__ = 'matches'
    id = Column(Integer, autoincrement=True, primary_key=True)
    Name = Column(Text)
    Odds = Column(Float)
    Type = Column(Text)
    Home = Column(Text)
    Away = Column(Text)
    Amount_got = Column(Float)
    Description = Column(Text)
    Certainity = Column(Float)
    Bookmaker = Column(Text)
    Match_result = Column(Text)
    Sport = Column(Text)
    League = Column(Text)
