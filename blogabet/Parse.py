import blogabet.model as model
import blogabet.sql as sql
import lxml
import blogabet.configuration


def extract_data(text, name):
    """
    Function that gets all the bets from blogabet site of NAME bettor as text and returns matches list
    """
    data = text.split('published a new pick:\n')
    df = []
    for dat in data[1:]:
        match = parseData(dat, name)
        if match is not None:
            df.append(match)
    return df


def home_away(line):
    """
    :param line: parse home and away team from line in form of: home -/v/vs away
    :return: home and away team names
    """
    full_match_name = line.split(" - ")
    if len(full_match_name) == 1:
        full_match_name = line.split(" v ")
        if len(full_match_name) == 1:
            full_match_name = line.split(" vs ")
            if len(full_match_name) == 1:
                home = ""
                away = ""
            else:
                home = full_match_name[0]
                away = full_match_name[1]
        else:
            home = full_match_name[0]
            away = full_match_name[1]
    else:
        home = full_match_name[0]
        away = full_match_name[1]
    return home, away


def certainity(line):
    """

    :param line: get certainity level from string of form 1/10
    :return: certainity of one bet as float
    """
    try:
        certainity = float(line[0:2])
    except ValueError:
        try:
            certainity = float(line[0])
        except ValueError:
            certainity = 0
    return certainity


def parse_different(text, name):
    """
    Same as parsedata funtion but takes text as list of strings not a single string
    :param text:
    :param name:
    :return: list of matches
    """
    match = model.Matches(Name=name)
    match.Home, match.Away = home_away(text[3])
    match.Type = text[4]
    try:
        match.odds = float(text[5][1:])
    except ValueError:
        return None
    match.Certainity = certainity(text[6])
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
    """
    Loads HTML files from database, parses matches and returns them as a list
    :return:  return list of matches
    """
    Session = sql.prepare_database(blogabet.configuration.database_url, model.Base)
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
            sql.update(Session, match)


def parseData(bet, name):
    """

    :param bet: single bet as a string from blogabet
    :param name: bettor name
    :return: matches object
    """
    if bet == '':
        return None
    data = bet.split("\n")
    match = model.Matches()
    match.Name = name
    match.Home, match.Away = home_away(data[1])
    try:
        match.Odds = float(data[2].split("@")[1])
    except ValueError:
        return None
    match.Type = data[2]
    hlp = data[3].split(" ")
    match.Certainity = certainity(data[3])
    # sometimes "LIVE" occurs before bookmaker name
    if hlp[1] == "LIVE":
        match.Bookmaker = hlp[2]
    else:
        match.Bookmaker = hlp[1]
    current_index = 2
    # there are some additional information that we need to skip(livestream available etc.)
    while (hlp[current_index][0] != '+') and (hlp[current_index][0] != '-'):
        current_index += 1
        if current_index >= len(hlp) or len(hlp[current_index]) == 0:
            return None
    match.Match_result = hlp[-1]
    pom = hlp[current_index].replace(",", "")
    match.Amount_got = float(pom)
    sport_row = data[4].split(" / ")
    match.Sport = sport_row[0]
    try:
        match.League = sport_row[1]
    except IndexError:
        match.League = ""
    match.Description = data[5]
    if match.Description == "Like":
        match.Description = ""
    return match
