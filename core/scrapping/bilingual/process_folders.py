import pandas as pd
import os
from core.datahandling import process_bilingual_data
from shutil import copyfile
import random


if __name__ == '__main__':



    path_img="/home/data/bilingual_manga/bi/"
    path="../../../data/bilingual_tsv"
    destination_dir="/home/data/bilingual_manga/bilingual_single_folder_images"
    destination_csv_dir="/home/data/bilingual_manga/"
    files:list=os.listdir(path)
    print(files)
    file_names:list=[i for i in files if ("selenium.tsv" in i)]

    full_path:str=[os.path.join(path,name) for name in files]

    mangas=process_bilingual_data.read_tsv_files(full_path)
    cuts=pd.qcut(mangas["font-size"],13,labels=False)
    mangas["font-size"]=cuts

    subgroup=mangas.groupby("manga")

    manga_names=mangas["manga"].unique()
    unique_img_id=0

    all_pds=[]
    type_of_analysis=["TRAIN","TRAIN","VALIDATE","TEST"]
    for manga_name in manga_names:

        relevant_group=subgroup.get_group(manga_name)
        page_numbers=relevant_group["id"].unique()
        unique_pages=relevant_group.groupby("id")
        for page_id in page_numbers:
            path_to_manga_dir=os.path.join(path_img,manga_name)
            path_to_file=os.path.join(path_to_manga_dir,"{}_jp.png".format(page_id))

            if(os.path.isfile(path_to_file)):
                process=process_bilingual_data.Preprocess_Bilingual()
                current_page_pd=unique_pages.get_group(page_id)
                process.set_data(current_page_pd)

                coords=process.to_box_coords(True)
                fonts=process.extract_font_size()

                if(pd.DataFrame(process.extract_font_size()).shape[0] <18):
                    full_unique_img_id="{}_jp.png".format(unique_img_id)
                    final_data=process.aggregate_to_pandas((coords,fonts))
                    final_data["id"]=unique_img_id

                    outside_params=final_data[(final_data.x1_jp > 1) | (final_data.x2_jp > 1) | (final_data.y1_jp > 1) | (final_data.x2_jp > 1)]



                    final_data["full_id"]='gs://typegan/single_folder_manga/{}'.format(full_unique_img_id)
                    final_data["original_path"]=path_to_file
                    final_data["eval_type"]= random.choice(type_of_analysis)

                    unique_img_id+=1
                    if len(outside_params) == 0:
                        all_pds.append(final_data)
                    #destination_file=os.path.join(destination_dir,full_unique_img_id)
                    #copyfile(path_to_file,destination_file)

    ordered_data_for_image_processing=pd.concat(all_pds,sort=False)
    print("finished")

    ordered_data_for_image_processing["gap1"]=""
    ordered_data_for_image_processing["gap2"] = ""
    ordered_data_for_image_processing["gap3"] = ""
    ordered_data_for_image_processing["gap4"] = ""
    ordered_data_for_image_processing["font-size_en"] = ordered_data_for_image_processing["font-size_en"].astype(int)

    formatted_data=ordered_data_for_image_processing[["eval_type","full_id", "font-size_en", "x1_en", "y1_en", "gap1", "gap2", "x2_en", "y2_en", "gap3", "gap4"]]
    formatted_data.to_csv(os.path.join(destination_csv_dir,"automl_format.csv"),index=False, header=False,index_label=False)

