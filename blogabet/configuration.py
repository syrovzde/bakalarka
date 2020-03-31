import blogabet.model

# for parsing name atm not used
introduction = "Hi, I'm "
# database url
database_url = "postgresql+pg8000://postgres:1234@localhost:5432/bettors"
# load older button on the page
button_xpath = '//*[@id="last_item"]/a'
# time to wait for element to load
wait = 8
#
declarative_base = blogabet.model.Base
# map of blogabet that contains all the subpages
sitemap = "https://blogabet.com/sitemap"
# element that contains all the bets
all = '/html/body/div[2]/section[2]/div[2]/div[1]/div[4]/ul/div/ul'
