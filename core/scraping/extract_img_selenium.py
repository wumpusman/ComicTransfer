from selenium import webdriver
import time

from PIL import Image
import os
import sys
from sklearn.neighbors import NearestNeighbors
import pandas as pd
sys.path.append("../")

from core.scraping import extract_img_selenium

def create_web_driver(path="chromedriver"):
    """
    creates a webdriver for selenium
    Args:
        path: path to chromedriver in selenium

    Returns:
        webdriver.driver
    """
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument("--window-size=4000,3000")
    return webdriver.Chrome(
        path,
        options=chrome_options)

def find_next_link(driver):
    """
    finds next link to go to for extracting image in manga
    Args:
        driver: chromedriver

    Returns:

    """
    buttons = driver.find_elements_by_class_name("btn-control-container")[0]
    buttons = buttons.find_elements_by_tag_name("a")
    if(len(buttons)<2): return []
    else:
        return [buttons[1].get_attribute("href")]


def save_eng_jp_pairs(driver, link, dir_path, file_id):
    """
    saves pairs of japanese and english pairs
    Args:
        driver: selenium driver
        link: link to page
        dir_path: path to save directory
        file_id: id of the file being saved

    Returns:
        None
    """
    if os.path.isdir(dir_path) == False:
        os.mkdir(dir_path)

    driver.get(link)
    time.sleep(1)
    path = os.path.join(dir_path, str(file_id))

    element = driver.find_element_by_class_name("image-container")

    save_image(driver, element, path + "_jp.png")
    driver.find_element_by_id("js-ripple-btn").click()
    time.sleep(.1)
    save_image(driver, element, path + "_en.png")

def save_image(driver, html_element, output):
    """
    save a screen show to the page
    Args:
        driver: driver for selenium
        html_element: the html to be saved
        output: output path

    Returns:
        None
    """
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



def align_jp_and_en_boxes(pd_results)->pd.DataFrame:
    """boxes are not ordered on the page, so heuristically must match them based on
    location on page
    """

    japanese_results = pd.DataFrame.copy(pd_results[pd_results.language == "jp"]).reset_index()
    english_results = pd.DataFrame.copy(pd_results[pd_results.language == "en"]).reset_index()

    japanese_vals = japanese_results[["left", "top"]].values
    english_vals = english_results[["left", "top"]].values

    n = NearestNeighbors(n_neighbors=1)
    n.fit((japanese_vals))
    dis, index = n.kneighbors(english_vals)
    english_results["boxID"] = index.reshape(-1)

    return japanese_results.append(english_results).reset_index()

def parse_color(key, string)->str:
    """
    parses the color html of form 'rgb(0,0,0)'
    Args:
        key: string that is a color
        string: associated string value in the html

    Returns:
        str
    """
    assert key == "color"
    string = string.replace("rgb(", "").replace(")", "").split(", ")
    return string


def parse_number(key, string)->float:
    """
    parse numeric values
    Args:
        key: type of html (currently ignored)
        string: a string that represents the font

    Returns:
        float
    """
    size = string.split(", ")[0].replace("px", "")
    return float(size)


def parse_known(key, val)->str:
    """
    maps string from html to to function for parsing
    Args:
        key: string from html
        val: associated value in html

    Returns:
        str
    """
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


def extract_dictionary(element)->dict:
    """
    extracts various pairings
    Args:
        element:

    Returns:
        dict
    """
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
        val = parse_known(key, val)
        dictionary[key] = val


    return dictionary


def save_meta_data_eng_jp_pairs(driver,link):
    """
    saves aggregate info about image such language and image size
    Args:
        driver: chrome driver used by selenium
        link: link to visit

    Returns:

    """
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
    """save image with both japanese text overlaid and english
    """
    if os.path.isdir(dir_path) == False:
        os.mkdir(dir_path)

    driver.get(link)
    time.sleep(1)
    path = os.path.join(dir_path, str(file_id))

    element = driver.find_element_by_class_name("image-container")

    save_image(driver, element, path + "_jp.png")
    driver.find_element_by_id("js-ripple-btn").click()
    time.sleep(.1)
    save_image(driver, element, path + "_en.png")