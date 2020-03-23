import blogabet.model as model


def extract_data(text, name):
    data = text.split('published a new pick:\n')
    df = []
    for dat in data[1:]:
        match = parseData(dat, name)
        if match is not None:
            df.append(match)
    return df


def home_away(line):
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
    try:
        certainity = float(line[0:2])
    except ValueError:
        try:
            certainity = float(line[0])
        except ValueError:
            certainity = 0
    return certainity


def parseData(bet, name):
    if bet == '':
        return None
    data = bet.split("\n")
    match = model.Matches()
    match.Name = name
    match.Home, match.Away = home_away(data[1])
    match.Odds = float(data[2].split("@")[1])
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
