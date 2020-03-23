import blogabet.model as model


def extract_data(text, name):
    data = text.split('published a new pick:\n')
    df = []
    for dat in data[1:]:
        match = parseData(dat, name)
        if match is not None:
            df.append(match)
    return df


def parseData(bet, name):
    if bet == '':
        return None
    data = bet.split("\n")
    match = model.Matches()
    match.Name = name
    full_match_name = data[1].split(" - ")
    if len(full_match_name) == 1:
        full_match_name = data[1].split(" v ")
        if len(full_match_name) == 1:
            full_match_name = data[1].split(" vs ")
            if len(full_match_name) == 1:
                match.Home = ""
                match.Away = ""
            else:
                match.Home = full_match_name[0]
                match.Away = full_match_name[1]
    match.Odds = float(data[2].split("@")[1])
    match.Type = data[2]
    hlp = data[3].split(" ")
    try:
        match.Certainity = float(data[3][0:2])
    except ValueError:
        match.Certainity = float(data[3][0])
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
