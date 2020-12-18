import urllib
from bs4 import BeautifulSoup as bs
from urllib.parse import urlencode, quote_plus, unquote
from urllib.request import urlopen, urlretrieve
import urllib
import os
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

def naver_crawling(find_namelist):
    for i in range(2):
        path = './MJK/data/image2/'+str(i+2)+'/'
        os.makedirs(path, exist_ok=True)
        driver = webdriver.Chrome('./Common_Code/chromedriver.exe')
        driver.get(
            "https://search.naver.com/search.naver?where=image&sm=tab_jum&query=")

        elem = driver.find_element_by_name("query")
        elem.send_keys(find_namelist[i+2])
        elem.send_keys(Keys.RETURN)

        SCROLL_PAUSE_TIME = 1
        # Get scroll height
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            # Scroll down to bottom
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)
            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:
                    driver.find_element_by_css_selector(".more_img").click()
                except:
                    break
            last_height = new_height

        # images = driver.find_elements_by_css_selector("._img")
        images = driver.find_elements_by_css_selector(".img_border")
        count = 1000
        for image in images:
            try:
                image.click()
                time.sleep(1)
                imgUrl = driver.find_element_by_xpath(
                    "/html/body/div[4]/div[2]/div[2]/div/a/img").get_attribute("src")
                opener = urllib.request.build_opener()
                opener.addheaders = [
                    ('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(imgUrl, path + str(count) + ".jpg")
                count = count + 1
            except:
                pass

if __name__ == '__main__':
    naver_crawling()