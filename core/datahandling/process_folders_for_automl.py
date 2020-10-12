
import os
import sys
from shutil import copyfile
import random
import argparse
import pandas as pd
sys.path.append("../")
from core.datahandling import process_bilingual_data

default_path_img_dir = "/home/data/bilingual_manga/bi/"
default_tsv_path = "../../data/bilingual_tsv"
default_destination_dir = "/home/data/bilingual_manga/bilingual_single_folder_images"
default_destination_csv = "/home/data/bilingual_manga/automl_format.csv"
project_bucket = "typegan/bilingual_single_folder_images/"

parser = argparse.ArgumentParser(
    description='setup for running and organizing files for automl')
parser.add_argument(
    "-i",
    "--imgdir",
    help="path to a jpg or png",
    default=default_path_img_dir)
parser.add_argument(
    "-t",
    "--tsvdir",
    help="path to font to be used",
    default=default_tsv_path)
parser.add_argument(
    "-d",
    "--destdir",
    help="path to destination folder",
    default=default_destination_dir)
parser.add_argument(
    "-c",
    "--destcsv",
    help="path to destination folder",
    default=default_destination_csv)
parser.add_argument(
    "-p",
    "--bucket",
    help="path where csv is to be placed",
    default=project_bucket)


def main(
        bucket_dir_path,
        img_directory_path: str,
        tsv_directory_path: str,
        destination_dir_path: str,
        destination_file_path: str):
    """
    script to format data for processing for google's automl image detection

    Args:
        bucket_dir_path: the path where the files will be finally stored
        img_directory_path: directory where all images are originally stored
        tsv_directory_path: directory where all tsv describing data is stored
        destination_dir_path: directory to save new asigned images
        destination_file_path: file to save formatted csv for automl

    Returns:
        None
    """

    files: list = os.listdir(tsv_directory_path)

    # the selenium files that have been extracted
    file_names: list = [i for i in files if ("selenium.tsv" in i)]

    full_path: str = [os.path.join(tsv_directory_path, name) for name in files]

    mangas = process_bilingual_data.read_tsv_files(full_path)
    # binarize fonts, not used, but neccesary for the model
    cuts = pd.qcut(mangas["font-size"], 13, labels=False)
    mangas["font-size"] = cuts

    subgroup = mangas.groupby("manga")

    manga_names = mangas["manga"].unique()
    unique_img_id = 0

    all_pds = []
    type_of_analysis = ["TRAIN", "TRAIN", "VALIDATE",
                        "TEST"]  # 50% train, 25% validate, 25% test
    for manga_name in manga_names:

        relevant_group = subgroup.get_group(manga_name)
        page_numbers = relevant_group["id"].unique()
        unique_pages = relevant_group.groupby("id")
        for page_id in page_numbers:
            path_to_manga_dir = os.path.join(img_directory_path, manga_name)
            path_to_file = os.path.join(
                path_to_manga_dir, "{}_jp.png".format(page_id))

            if (os.path.isfile(path_to_file)):
                process = process_bilingual_data.Preprocess_Bilingual()
                current_page_pd = unique_pages.get_group(page_id)
                process.set_data(current_page_pd)

                coords = process.to_box_coords(True)
                fonts = process.extract_font_size()

                if (pd.DataFrame(process.extract_font_size()
                                 ).shape[0] < 18):  # automl has a cap of 20
                    full_unique_img_id = "{}_jp.png".format(unique_img_id)
                    final_data = process.aggregate_to_pandas((coords, fonts))
                    final_data["id"] = unique_img_id

                    outside_params = final_data[(final_data.x1_jp > 1) | (
                        final_data.x2_jp > 1) | (final_data.y1_jp > 1) | (final_data.x2_jp > 1)]

                    final_data["full_id"] = 'gs://{}/{}'.format(
                        bucket_dir_path, full_unique_img_id)
                    final_data["original_path"] = path_to_file
                    final_data["eval_type"] = random.choice(type_of_analysis)

                    unique_img_id += 1
                    if len(outside_params) == 0:  # no errors
                        all_pds.append(final_data)
                    destination_file = os.path.join(
                        destination_dir_path, full_unique_img_id)
                    copyfile(destination_dir_path, destination_file)

    ordered_data_for_image_processing = pd.concat(all_pds, sort=False)
    print("finished")

    ordered_data_for_image_processing["gap1"] = ""
    ordered_data_for_image_processing["gap2"] = ""
    ordered_data_for_image_processing["gap3"] = ""
    ordered_data_for_image_processing["gap4"] = ""
    ordered_data_for_image_processing["font-size_en"] = ordered_data_for_image_processing["font-size_en"].astype(
        int)

    formatted_data = ordered_data_for_image_processing[[
        "eval_type", "full_id", "font-size_en", "x1_en", "y1_en", "gap1", "gap2", "x2_en", "y2_en", "gap3", "gap4"]]
    formatted_data.to_csv(destination_file_path, index=False, header=False,
                          index_label=False)


if __name__ == '__main__':

    args = parser.parse_args()

    main(args.bucket, args.imgdir, args.tsvdir, args.destdir, args.destcsv)
