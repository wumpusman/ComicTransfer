import os
import pandas as pd
import sys
import time
import pandas as pd
import argparse
sys.path.append("../")
from core.scraping import extract_img_selenium

# was tested originally on windows
driver_path = "downloads/chromedriver_win32//chromedriver"
save_dir = "../data/bilingual_save"
manga_files_path = "../data/manga_list.txt"
max_pages = 3

parser = argparse.ArgumentParser(description='running extraction code')
parser.add_argument("-d",
                    "--driver_path",
                    help="path to chrome driver",
                    default=driver_path)
parser.add_argument("-s",
                    "--save_dir",
                    help="path to save files and images",
                    default=save_dir)
parser.add_argument(
    "-m",
    "--manga_list",
    help=
    "path to list of urls with each line referencing a link to one page per different manga",
    default=manga_files_path)
parser.add_argument("-n",
                    "--number_per_manga",
                    help="max number of pages to attempt to save per page",
                    default=int(max_pages),
                    type=int)

# python scraping/main_extraction.py -m "../data/manga_list.txt"
# --number_per_manga 3


def main(driver_path: str,
         save_dir: str,
         manga_list_pth: str,
         max_pages_per_manga=1000):
    """
    used to extract manga meta data and images specified in in the mangalist each line refers to a page in a different manga
    Args:
        driver_path: path to selenium driver
        save_dir: directory to save all the mangas images
        manga_list_pth: path to a text file where each line is a different manga link
        max_pages_per_manga: cap how many pages you extract per manga

    Returns:
        None
    """

    driver = extract_img_selenium.create_web_driver(driver_path)
    f = open(manga_list_pth, "r")
    manga_list = f.readlines()

    for manga_initial_page in manga_list:

        current_link = manga_initial_page.replace("\n", "")
        manga_name = manga_initial_page.replace("https://", "").split("/")[2]
        current = 0
        all_frames = pd.DataFrame()

        while current < max_pages_per_manga:  # max cap on links
            pd_results = None
            driver.get(current_link)
            try:  # not all pages are extractable, so just skip them
                results = extract_img_selenium.save_meta_data_eng_jp_pairs(
                    driver, current_link)
                page_height = driver.find_elements_by_class_name(
                    "image-container")[0].size["height"]
                page_width = driver.find_elements_by_class_name(
                    "image-container")[0].size["width"]

                pd_results = pd.DataFrame(results)
                pd_results["manga"] = manga_name
                pd_results["page_width"] = page_width
                pd_results["page_height"] = page_height
                pd_results = extract_img_selenium.align_jp_and_en_boxes(
                    pd_results)
                pd_results["link"] = current_link
                img_number = current
                # what image is this associated with in the drive
                pd_results["id"] = img_number
                all_frames = all_frames.append(pd_results)
                full_path = os.path.join(save_dir, manga_name)
                extract_img_selenium.save_eng_jp_pairs(driver, current_link,
                                                       full_path, img_number)

            except BaseException:
                print("problem at {}".format(current_link))

            current += 1

            next_link_ary = extract_img_selenium.find_next_link(driver)

            if len(next_link_ary) == 0 or current >= max_pages_per_manga:
                all_frames = all_frames.drop("level_0", axis=1).reset_index()
                tsv_to_save = os.path.join(save_dir,
                                           "{}{}".format(manga_name, ".tsv"))
                all_frames.to_csv(tsv_to_save, sep="\t")
                break

            current_link = next_link_ary[0]


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.driver_path, args.save_dir, args.manga_list,
         args.number_per_manga)
