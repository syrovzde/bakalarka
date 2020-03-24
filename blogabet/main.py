import pandas

import blogabet.crawler as crawler
import lxml.html
import blogabet.sql as dtb
import blogabet.configuration as configuration
import blogabet.model as model
import blogabet.Parse
import os
import re

"""

"""

def crawl():
    df = pandas.read_csv("updated_list.csv")
    crawler.crawl(df['url'], df['name'])



if __name__ == '__main__':
    crawl()
