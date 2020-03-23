import pandas

import blogabet.crawler as crawler


def crawl():
    df = pandas.read_csv("updated_list.csv")
    crawler.crawl(df['url'], df['name'])


"""Session = dtb.prepare_database(configuration.database_url,model.Base)
data = Session.query(model.Bettor).limit(1).all()
for dat in data:
    tree = lxml.html.document_fromstring(dat.HTML)
    i = 0
    for element in tree.cssselect('div.media-body'):
        if i == 15:
            print(str(element.text_content()).replace(" ","").replace("\n\n",""))
            #print(element.cssselect('div.pick-line')[0].text_content())
            #print(str((element.cssselect('div.labels')[0].text_content())).replace(" ",""))
            #print(element.cssselect('div.sport-line')[0].text_content())
            #print(element.cssselect('div.col-xs-12._text-more.feed-analysis')[0].text_content())
        i += 1"""
crawl()
