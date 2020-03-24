def drop_wrong_urls(df):
    """drop obviously wrong urls"""
    df = df.dropna()
    return df[df.url.str.contains(".com")]
