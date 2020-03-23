def drop_wrong_urls(df):
    df = df.dropna()
    return df[df.url.str.contains(".com")]
