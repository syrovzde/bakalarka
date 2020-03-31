from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy import MetaData

#
url = "postgresql+pg8000://postgres:1234@localhost:5432/postgres"
m = MetaData(schema="BetExplorer")
Base = automap_base(metadata=m)
engine = create_engine(url)

Base.prepare(engine, reflect=True)
bettor_html = Base.classes.Matches
session = Session(engine)

pom = session.query(Base.classes.Matches, Base.classes.Odds_1x2).join(Base.classes.Matches,
                                                                      Base.classes.Matches.MatchID == Base.classes.Odds_1x2.MatchID).all()
print(pom[10][1].Bookmaker)
