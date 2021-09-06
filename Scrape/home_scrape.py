import sys
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from home_info import *


# ======================
# Clean a price string
# ======================
def cleanPrice(price):
    price = price.replace('€', '')
    price = price.replace(' ', '')
    price = price.replace(',', '')
    price = price.replace('.', '')
    return price


# ======================
# Clean a date string
# ======================
def cleanDate(date):
    try:
        day, month, year = date.split('/')
        return '%04d-%02d-%02d' % (int(year), int(month), int(day))
    except:
        return date


# =========================
# Get all data for a home
# =========================
def getHomeInfo(homeUrl):
    try:
        TOKEN = 'QGNY3bJnz8Ve12BBVqGBFg'
        response = requests.get('https://api.proxycrawl.com/?token=%s&url=%s' % (TOKEN, quote(homeUrl)))
    except:
        return
    if response.content == None:
        return
    soup = BeautifulSoup(response.content, features='html.parser')

    mainContent = soup.find('div', attrs={'id': 'listingDetailsMainContent'})
    if mainContent == None:
        return
    info = mainContent.find('div', attrs={'class': 'padding-phone-only'})
    if info == None:
        return

    homeInfo = HomeInfo()
    homeInfo.url = homeUrl

    homeElements = info.find_all('div', attrs={'class': 'padding-small-top'})
    if homeElements == None:
        return

    for homeElement in homeElements:
        key = homeElement.find('div', attrs={'class': 'desktop-3-details'}).text.strip()
        if (key == 'Price'):
            priceContainer = homeElement.find('div', attrs={'class': 'desktop-5-details'})
            if priceContainer == None:
                return
            value = priceContainer.find('span', attrs={'class': 'padding-right'})
        else:
            value = homeElement.find('div', attrs={'class': 'desktop-5-details'})

        value = value.text.strip().replace('\n', ' ').replace('"', '')

        fieldLabel = homeInfo.fieldLabelsEn.get(key)

        if fieldLabel != None:
            setattr(homeInfo, fieldLabel, value)

    sideElements = soup.find('div', attrs={'id': 'carouselSideDetails'}).find_all(
        'div', attrs={'class': 'tableRow'}
    )
    if sideElements != None:
        for sideElement in sideElements:
            cells = sideElement.find_all('div', attrs={'class': 'tableCell'})
            key = cells[0].text.strip()
            value = cells[1].text.strip()
            if (key == 'Neighborhood'):
                homeInfo.generalRegion = value
                break

    # Clean prices
    homeInfo.price = cleanPrice(homeInfo.price)
    homeInfo.pricePerSqm = cleanPrice(homeInfo.pricePerSqm)

    # Clean dates
    homeInfo.availableDate = cleanDate(homeInfo.availableDate)
    homeInfo.lastUpdate = cleanDate(homeInfo.lastUpdate)
    f.write(str(homeInfo) + '\n')
    c.incr()
    print(c.get_count(), " properties downloaded")


# =========================
# Get all homes on a page
# =========================
def getPageHomes(pageUrl):
    print('------> Getting %s...' % pageUrl)

    TOKEN = 'QGNY3bJnz8Ve12BBVqGBFg'
    response = requests.get('https://api.proxycrawl.com/?token=%s&url=%s' % (TOKEN, quote(pageUrl)))
    soup = BeautifulSoup(response.content, features='html.parser')

    listings = soup.find_all('div', attrs={'class': 'searchListing'})
    if listings == None:
        return
    for listing in listings:
        urls = listing.find('div', attrs={'class': 'bd'}).find_all('a')
        if urls == None:
            return
        homeUrl = urls[1]['href']
        getHomeInfo(homeUrl)

    return soup


# ============================
# Get all homes for a region
# ============================
def getRegionHomes(regionUrl):
    TOKEN = 'QGNY3bJnz8Ve12BBVqGBFg'
    response = requests.get('https://api.proxycrawl.com/?token=%s&url=%s' % (TOKEN, quote(regionUrl)))
    soup = BeautifulSoup(response.content, features='html.parser')

    getPageHomes(regionUrl)
    nextPageElement = soup.find('li', attrs={'class': 'next'})
    if nextPageElement == None:
        return

    while nextPageElement.find('a', attrs={'class': 'disable'}) == None:
        if c.get_count() >= 50:
            return
        # There is a next page; find its URL
        pageUrl = nextPageElement.find('a')['href']
        soup = getPageHomes(pageUrl)
        nextPageElement = soup.find('li', attrs={'rel': 'next'})
        if nextPageElement == None:
            return


# ===============
# Get all homes
# ===============
def getAllHomes(url):
    TOKEN = 'QGNY3bJnz8Ve12BBVqGBFg'
    response = requests.get('https://api.proxycrawl.com/?token=%s&url=%s' % (TOKEN, quote(url)))
    soup = BeautifulSoup(response.content, features='html.parser')

    regionContainer = soup.find('div', attrs={'id': 'latestTabs_popularSearches'})
    if regionContainer == None:
        return
    regionElements = regionContainer.find_all('div', attrs={'class': 'css-table-cell padding'})
    if regionElements == None:
        return

    for regionElement in regionElements:
        if c.get_count() >= 50:
            return
        if regionElement.find('span', attrs={'class': 'orange-text'}) == None:
            # Only pick URLs for region names, not number of homes (orange numbers)
            regionUrl = regionElement.find('a')['href']
            getRegionHomes(regionUrl)


class counter:
    def __init__(
            self,
            count=0
    ):
        self.count = count

    def incr(self):
        self.count += 1

    def get_count(self):
        return self.count


# ======
# Main
# ======
if __name__ == '__main__':
    c = counter(0)
    # to enable command line access have check to ensure the poper num of arguments is entered

    if len(sys.argv) >= 2:
        validPropertyTypes = ['residential', 'commercial', 'land']
        propertyType = sys.argv[1]
        if propertyType not in validPropertyTypes:
            print('Usage: %s <propertType (%s)>' % (sys.argv[0], '|'.join(validPropertyTypes)))
            sys.exit()
    else:
        propertyType = 'residential'

    date = datetime.now().strftime('%Y%m%d')

    # created a file to store he data to check the quality before uploading to database

    f = open('properties-%s-%s.txt' % (propertyType, date), 'w', encoding='utf-8', buffering=1)

    # create object to store field information from site

    homeInfo = HomeInfo()

    # Get field names and write header to file

    fields = [field for field in dir(homeInfo)
              if not field.startswith('__') and
              field != 'fieldLabelsGr' and
              field != 'fieldLabelsEn' and
              field != 'self']
    f.write('~'.join(fields) + '\n')

    # Get all homes for sale
    if propertyType == 'residential':
        url = 'https://en.spitogatos.gr/sale'
    else:
        url = 'https://en.spitogatos.gr/sale/' + propertyType

    getAllHomes(url)
