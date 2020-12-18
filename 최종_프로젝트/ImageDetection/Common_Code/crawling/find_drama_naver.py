from bs4 import BeautifulSoup as bs
from urllib.parse import urlencode, quote_plus, unquote
from urllib.request import urlopen, urlretrieve
import urllib
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import naver_crawling, google_crawling
import os
  
base_url = 'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query='
plusUrl = input('검색어 입력: ')
url = base_url + quote_plus(plusUrl) + '%EC%B6%9C%EC%97%B0%EC%A7%84'

html = urlopen(url)
soup = bs(html, "html.parser")
name = soup.find("div", class_="list_image_info _content").find_all("li")

find_imglist = list()
find_casting = list()
find_namelist = list()

for item in name:
    find_name = item.find_all(class_="_text")[1]  # 주인공 이름
    find_namelist.append(find_name.get_text())

    find_img = item.find(class_='item').find_all(class_='thumb')
    for j in find_img:
        img = j.find('img')
        find_imglist.append(img.get('src'))
        find_casting.append(img.get('alt'))

find_imglist = np.array(find_imglist)
find_casting = np.array(find_casting)
find_namelist = np.array(find_namelist)

google_crawling.google_crawling(find_namelist)
naver_crawling.naver_crawling(find_namelist)




# np.save('./Common_data/npy/find_imglist.npy', arr=find_imglist)
# np.save('./Common_data/npy/find_casting.npy', arr=find_casting)
# np.save('./Common_data/npy/find_namelist.npy', arr=find_namelist)

