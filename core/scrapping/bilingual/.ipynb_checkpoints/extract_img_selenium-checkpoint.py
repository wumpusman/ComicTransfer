from selenium import webdriver
import time

from PIL import Image
import os
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def create_web_driver(path="chromedriver"):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--headless')
    #chrome_options.add_argument("--window-size=4000,3000")
    return webdriver.Chrome(
        path,
        options=chrome_options)


def save_image(driver, html_element, output):
    driver.save_screenshot(output)

    height = html_element.get_attribute("height")
    width = html_element.get_attribute("width")
    location = html_element.location
    size = html_element.size

    x = location['x']
    y = location['y']
    width = location['x'] + size['width']
    height = location['y'] + size['height']

    im = Image.open(output)
    im = im.crop((int(x), int(y), int(width), int(height)))
    im.save(output)



def align_jp_and_en_boxes(pd_results):
    japanese_results = pd.DataFrame.copy(pd_results[pd_results.language == "jp"]).reset_index()
    english_results = pd.DataFrame.copy(pd_results[pd_results.language == "en"]).reset_index()

    japanese_vals = japanese_results[["left", "top"]].values
    english_vals = english_results[["left", "top"]].values

    n = NearestNeighbors(n_neighbors=1)
    n.fit((japanese_vals))
    dis, index = n.kneighbors(english_vals)
    english_results["boxID"] = index.reshape(-1)

    return japanese_results.append(english_results).reset_index()

def parse_color(key, string):
    assert key == "color"
    string = string.replace("rgb(", "").replace(")", "").split(", ")
    return string


def parse_number(key, string):
    size = string.split(", ")[0].replace("px", "")
    return float(size)


def parseKnown(key, val):
    key_to_func = {}
    key_to_func["left"] = parse_number
    key_to_func["top"] = parse_number
    key_to_func["width"] = parse_number
    key_to_func["font-size"] = parse_number
    key_to_func["color"] = parse_color

    if key in key_to_func:
        return key_to_func[key](key, val)
    else:
        return val


def extract_dictionary(element):
    all_pairs = []
    index = 0

    style = element.get_attribute("style")
    dictionary = {}
    attribute_elements = style.split(";")
    for element in attribute_elements:
        pair = element.split(":")
        if len(pair) != 2:
            continue
        key = pair[0].replace(" ", "")
        val = pair[1]
        val = parseKnown(key, val)
        dictionary[key] = val

    # dictionary.update(additional_dict_values)

    return dictionary


def save_meta_data_eng_jp_pairs(driver,link):
    jp_ones = driver.find_elements_by_class_name("main")[0].find_elements_by_class_name("div-text")
    en_ones = driver.find_elements_by_class_name("sub")[0].find_elements_by_class_name("div-text")
    counterJp=0
    counterEn=0
    full_list:list=[]
    for element in jp_ones:
        primary:dict=extract_dictionary(element)
        text:str=element.text
        innerWidth:int=element.size["width"]
        innerHeight:int=element.size["height"]
        primary["language"] = "jp"
        primary["text"] = text
        primary["innerWidth"]=innerWidth
        primary["innerHeight"]=innerHeight
        primary["link"]=link
        primary["boxID"]=counterJp
        counterJp+=1
        full_list.append(primary)

    driver.find_element_by_id("js-ripple-btn").click()
    time.sleep(.1)
    for element in en_ones:
        primary: dict = extract_dictionary(element)
        text: str = element.text
        innerWidth: int = element.size["width"]
        innerHeight: int = element.size["height"]
        primary["language"] = "en"
        primary["text"] = text
        primary["innerWidth"] = innerWidth
        primary["innerHeight"] = innerHeight
        primary["link"] = link
        primary["boxID"] = counterEn
        counterEn += 1
        full_list.append(primary)
    return full_list




def save_eng_jp_pairs(driver, link, dir_path, file_id):
    if os.path.isdir(dir_path) == False:
        os.mkdir(dir_path)

    driver.get(link)
    time.sleep(1)
    path = os.path.join(dir_path, file_id)

    element = driver.find_element_by_class_name("image-container")

    save_image(driver, element, path + "_jp.png")
    driver.find_element_by_id("js-ripple-btn").click()
    time.sleep(.1)
    save_image(driver, element, path + "_en.png")