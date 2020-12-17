#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 09:21:57 2020

@author: Han
"""


from selenium import webdriver

# 1. chrome driver options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')


# 2. 라이브러리 import
import os
import time
import socket

from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError # 홈페이지와 검색결과에러찾기 
from selenium.common.exceptions import ElementClickInterceptedException, NoSuchElementException, ElementNotInteractableException
from PIL import Image


# 3. 스크래핑 실행 
socket.setdefaulttimeout(30) # 너무 오래걸릴 경우 사용


# scroll down () : 스크롤을 내리는 함수
def scroll_down():
    scroll_count = 0
    print("scroll down() : 스크롤 다운 시작")
    
    last_height = wd.execute_script("return document.body.scrollHeight")
    after_click = False 
    
    while True : 
        print(f"스크롤 다운 {scroll_count}")
        wd.execute_script("window.scrollTo(0,document.body.scrollHeight);")
        scroll_count += 1 
        time.sleep(1) # 대기
        
        new_height = wd.execute_script("return document.body.scrollHeight")
        
        
        if last_height == new_height: # scroll down이 실행되지 않았다
            if after_click is True:
                break
            
            else:
                try :
                    more_button = wd.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input')
                
                
                    if more_button.is_displayed():
                        more_button.click()
                        after_click = True
                        
                except NoSuchElementException as e :
                        print("exception is ", e)
                        break
            
        last_height = new_height
        

# click and save() : thumbnail  이미지 선택후 원본 이미지 저장 
def click_and_save(dir_name,index,img, img_list_length):
    global scrapped_count
    
    try:
        img.click()
        wd.implicitly_wait(3) # 클릭후 대기시간 3초
        src = wd.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img').get_attribute('src')
        
        if src.split('.')[-1] == 'png':
            urlretrieve(src,dir_name + '/' + str(scrapped_count+1)+ ".png")
            print(f"{index+1}/{img_list_length} PNG 이미지 저장")
        else:
            urlretrieve(src,dir_name + '/' + str(scrapped_count+1) + ".jpg")
            print(f"{index+1}/{img_list_length} jpg 이미지 저장")
        scrapped_count += 1
        
    except HTTPError as e:
        print(e)
        pass
    except ElementClickInterceptedException as e:
        print(e)
        wd.execute_script("window.scrollTo(0,window.scrollY + 100")
        wd.sleep(1)
        click_and_save(dir_name, index, img,len(img_list))
        
    
# filter_and_remove : 일정 해상도 이하이거나 손상된 이미지 제거
def filter_and_remove(dir_name,query,filter_size):
    filtered_count = 0
    
    for index, file_name in enumerate(os.listdir(dir_name)):
        file_path = os.path.join(dir_name,file_name)
        img = Image.open(file_path)
        try:
            if img.width < filter_size and img.height < filter_size :
                img.close()
                os.remove(file_path)
                print(f"{index} 이미지 제거")
                filtered_count += 1
        except OSError as e:
            print(e)
            os.remove(file_path)
            filtered_count += 1
    print(f"이미지 제거 개수 : {filtered_count}/{scrapped_count}")
        
    
        
        
        
# scraping 함수 정의 : 구글 이미지 스크래핑 함수
def scraping(dir_name,query):
    global scrapped_count
    
    url = f"https://www.google.com/search?q={query}&tbm=isch&ved=2ahUKEwi4pYDrptHtAhVKAKYKHbfJA6gQ2-cCegQIABAA&oq=sea&gs_lcp=CgNpbWcQAzIFCAAQsQMyAggAMgUIABCxAzICCAAyAggAMgUIABCxAzICCAAyAggAMggIABCxAxCDATICCABQ1w5Y9hBgwhRoAHAAeACAAXyIAeMCkgEDMC4zmAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=nFzZX7jnEsqAmAW3k4_ACg&bih=721&biw=1280"
    wd.get(url)
    wd.maximize_window()
    
    scroll_down() # 스크롤 다운
    
    div = wd.find_element_by_xpath('//*[@id="islrg"]/div[1]')
    img_list = div.find_elements_by_css_selector('div.bRMDJf.islir > img')
    
    for index, img in enumerate(img_list):
        
        try:
            click_and_save(dir_name, index, img,len(img_list)) # 이미지 다운로드
            
            
            
        except ElementClickInterceptedException as e:
            print(e)
            wd.execute_script("window.scrollTo(0,window.scrollY + 100")
            wd.sleep(1)
            click_and_save(dir_name, index, img,len(img_list))
        
        except NoSuchElementException as e:
            print(e)
            wd.execute_script("window.scrollTo(0,window.scrollY + 100")
            wd.sleep(1)
            click_and_save(dir_name, index, img,len(img_list))
        
        except ConnectionRefusedError as e:
            print(e)
            pass
        
        except URLError as e:
            print(e)
            pass
        
        except socket.error as e:
            print(e)
            pass
        
        except socket.gaierror as e:
            print(e)
            pass
        
        except ElementNotInteractableException as e:
            print(e)
            break
    try:
        
        print("스크래핑 종류 (성공률 : %.2f%%)"%(scrapped_count/ len(img_list) * 100.0))
    except ZeroDivisionError as e:
        print(e)
    
    wd.quit()
    
        



# driver 홈페이지 접근 (각자 드라이버 path설정)
wd = webdriver.Chrome("/Users/Han/Download2/chromedriver",options=chrome_options)

scrapped_count = 0 # 갯수 초기화
# 이미지 저장 path 설정 
path = "/Users/Han/Download2/BigdataGroupProject/BigdataGroupProject/images/" 
query = input("검색어 입력 : ")

dir_name = path + query
os.makedirs(dir_name)
print(f"{dir_name}디렉토리 생성")

scraping(dir_name, query)
filter_and_remove(dir_name,query,400)










