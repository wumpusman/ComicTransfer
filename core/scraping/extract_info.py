from extract_img_selenium import *
from bs4 import BeautifulSoup
import requests as req
import pandas as pd
from core.scraping import extract_img_selenium
import urllib

import os
import time


def extract_dictionary(element, additional_dict_values: dict = {}):
    """
    extracts key value pairs from html parsed by beautiful soup of bilingualmagna
    Args:
        element:
        additional_dict_values:

    Returns:

    """
    all_pairs = []
    index = 0
    for sub in element:
        style = sub.get_attribute_list("style")

        dictionary = {}
        attribute_elements = style[0].split(";")
        for element in attribute_elements:

            pair = element.split(":")
            if len(pair)!=2:
                continue
            key = pair[0]
            val = pair[1]
            val = extract_img_selenium.parse_known(key, val)
            dictionary[key] = val
        dictionary["text"] = sub.getText().replace(",", "")
        dictionary["box_id"] = index
        index += 1
        dictionary.update(additional_dict_values)
        all_pairs.append(dictionary)

    return all_pairs


def parse_soup_page(soup, link=""):
    link = ""
    jp = soup.select("div.main.language")[0]
    eng = soup.select('div.language.sub')[0]
    eng_text = eng.findAll("div", "div-text")
    jp_text = jp.findAll("div", "div-text")

    sub = soup.findAll(False, {'class': ['image-container']})[0]
    img_link = sub.find("img")
    jpg = img_link.attrs["src"]

    manga = soup.findAll(False, {'id': ["manga-name"]})[0]
    manga = manga.text.replace(" ", "_").replace("?","").replace(":","")
    print(manga)

    jp_dict = {}
    eng_dict = {}

    eng_dict = extract_dictionary(eng_text, {"language": "english", "manga": manga})

    jp_dict = extract_dictionary(jp_text, {"language": "jp", "manga": manga})

    translate = pd.DataFrame(jp_dict)
    translate = translate.append(eng_dict, True)
    translate["image"] = jpg
    translate["link"] = link

    next_page_href = ""
    next_link = soup.select('div.btn-control-container')
    next_link = next_link[0].select('a')
    if len(next_link) > 1:
        next_page_href = next_link[1]["href"]

    return translate


def save_file(dir_path, file_id, link):
    if os.path.isdir(dir_path) == False:
        os.mkdir(dir_path)

    path = os.path.join(dir_path, file_id)

    urllib.request.urlretrieve(link, path + ".jpg")


def get_link(soup):
    next_page_href = ""
    next_link = soup.select('div.btn-control-container')
    next_link = next_link[0].select('a')
    if len(next_link) > 1:
        next_page_href = next_link[1]["href"]

    return next_page_href


def extract_manga(link, save_dir="/home/data/bilingual/"):
    all_frames = pd.DataFrame()
    for i in range(3000):
        last_link = link
        resp = req.get(link)
        soup = BeautifulSoup(resp.text, 'lxml')

        link = get_link(soup)  # link to next page
        if link =="":
            break
        time.sleep(1)
        frame = []
        try:
            frame = parse_soup_page(soup, link)

        except:
            pass

        link = "https://bilingualmanga.com/" + link

        print(link)
        print(len(frame))
        if len(frame) != 0:
            frame["id"] = i
            frame["link"] = last_link
            all_frames = all_frames.append(frame)

            img_link = frame["image"].unique()[0]
            manga_name = frame["manga"].unique()[0]

            manga_dir = os.path.join(save_dir, manga_name)
            print(img_link)

            save_file(manga_dir, str(i), img_link)

    all_frames = all_frames.reset_index()
    return all_frames