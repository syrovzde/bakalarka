import time

import math
import selenium
from selenium import webdriver

import blogabet.Parse as parse
import blogabet.configuration as configuration
import blogabet.model as model
import blogabet.sql as db


def get_chromewebdriver(chrome_options=None):
    """

    :param chrome_options: specified options to driver
    :return:configured chrome driver for selenium
    """
    if chrome_options is None:
        chrome_options = selenium.webdriver.chrome.options.Options()
        chrome_options.add_argument("--headless")
    browser = webdriver.Chrome(options=chrome_options)
    if configuration.wait > 0:
        browser.implicitly_wait(configuration.wait)
    return browser


def crawl(urls, names):
    """

    :param urls:list of url to crawl
    :param names:  list of names corresponding to urls
    Get Html from site parses it and writes to database
    """
    browser = get_chromewebdriver(chrome_options=None)
    Session = db.prepare_database(configuration.database_url, configuration.declarative_base)
    for url, name in zip(urls, names):
        exists = Session.query(model.Bettor).filter_by(name=name).scalar() is not None
        if exists:
            print("existed")
            continue
        html, text = crawl_url(browser=browser, url=url)
        if html is not None and text is not None:
            matches = parse.extract_data(text, name)
            db.update(Session, model.Bettor(name=name, HTML=html, url=url))
            for match in matches:
                db.update(Session, match)
    Session.close()


def crawl_url(url, browser):
    """

    :param url: url to crawl
    :param browser:  driver for selenium
    :return HTML as whole page content
    :return bet list from the page as text
    """
    browser.get(url)
    try:
        picks = int(browser.find_element_by_xpath(xpath='// *[ @ id = "header-picks"]').text)
    except selenium.common.exceptions.NoSuchElementException:
        print("data cannot be loaded")
        return None, None
    iteration_count = math.ceil((picks - 10) / 10.0)
    i = 1
    while True:
        try:
            button = browser.find_element_by_xpath(xpath='//*[@id="last_item"]/a')
        except:
            print("ended")
            break
        x = button.location['x']
        y = button.location['y']
        browser.execute_script("window.scrollTo(arguments[0],arguments[1])", x, y - 70)
        button.click()
        time.sleep(1.5)
        print("Iteration {i} out of {count} finished".format(i=i, count=iteration_count))
        i += 1

    html = browser.page_source
    try:
        text = browser.find_element_by_xpath(xpath='/html/body/div[2]/section[2]/div[2]/div[1]/div[4]/ul/div/ul').text
    except selenium.common.exceptions.NoSuchElementException:
        text = None

    return html, text
