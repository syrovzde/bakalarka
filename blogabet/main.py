import pandas

import blogabet.crawler as crawler
import lxml.html
import blogabet.sql as dtb
import blogabet.configuration as configuration
import blogabet.model as model
import blogabet.Parse
import os
import re
def crawl():
    df = pandas.read_csv("updated_list.csv")
    crawler.crawl(df['url'], df['name'])


def parse_different(text, name):
    match = model.Matches(Name=name)
    match.Home, match.Away = blogabet.Parse.home_away(text[3])
    match.Type = text[4]
    try:
        match.odds = float(text[5][1:])
    except ValueError:
        return None
    match.Certainity = blogabet.Parse.certainity(text[6])
    amount_found = False
    bookmaker_found = False
    i = 7
    for cell in text[7:]:
        if cell != "LIVE" and cell != 'i' and not bookmaker_found:
            match.Bookmaker = cell
            i += 1
            bookmaker_found = True
            continue
        if ("+" in cell or "-" in cell or '0' in cell) and not amount_found:
            match.Amount_got = float(cell)
            i += 1
            break
        i += 1
    match.Match_result = text[i + 1]
    i += 1
    sport_line = text[i].split(" / ")
    match.Sport = sport_line[0]
    if len(sport_line) != 1:
        match.League = sport_line[1]
    else:
        match.League = ""
    i += 2
    match.Description = text[i]
    if match.Description == "LIKE":
        match.Description = ""
    return match


def data_from_html():
    Session = dtb.prepare_database(configuration.database_url, model.Base)
    data = Session.query(model.Bettor).all()
    for dat in data:
        tree = lxml.html.document_fromstring(dat.HTML)
        matches = []
        i = 0
        pom = tree.cssselect('div.media-body')
        for element in tree.cssselect('div.media-body'):
            reduced = []
            asstr = str(element.text_content()).splitlines()
            for s in asstr:
                tmp = s.strip()
                if tmp != "":
                    reduced.append(tmp)
            match = parse_different(reduced, dat.name)
            if match is not None:
                matches.append(match)
            print("iteration {i} out of {j}".format(i=i, j=len(pom)))
            i += 1
        for match in matches:
            dtb.update(Session, match)


if __name__ == '__main__':
    crawl()
