from core.components.translation import predictor_translate
from core.datahandling import process_bilingual_data


import argparse
import os


if __name__ == '__main__':

    translator_obj=predictor_translate.TranslationGoogle("typegan")
    tsv_dir = "/home/jupyter/ComicTransfer2/ComicTransfer/data/bilingual_tsv"
    files:list=os.listdir(tsv_dir)

    file_names: list = [os.path.join(tsv_dir,i) for i in files if ("selenium.tsv" in i)]
    mangas = process_bilingual_data.read_tsv_files(file_names)

    column_names = mangas.columns

    process_obj=process_bilingual_data.Preprocess_Bilingual()
    process_obj.set_data(mangas)

    process_obj._data_to_process
    column_names = process_obj._data_to_process

    org_pages=process_obj._data_to_process.groupby("link")
    individual_links=process_obj._data_to_process["link"].unique()
    print("OK")

    specific_page = org_pages.get_group(individual_links[4])
    japanese_text=specific_page.text_jp

    translator_obj=predictor_translate.TranslationGoogle("typegan")
    print("OK")

    specific_page = org_pages.get_group(individual_links[10])
    translations = translator_obj.predict(specific_page.text_jp.str.replace("\r\n", " "))
    import sklearn
    from sklearn.neighbors import NearestNeighbors
    import pandas as pd
    temp_process=process_bilingual_data.Preprocess_Bilingual()

    temp_process._data_to_process=specific_page

    coords=pd.DataFrame(process_obj.to_box_coords())

    nn=NearestNeighbors(n_neighbors=3)
    distances=coords[["x1_jp","y1_jp","x2_jp","y2_jp"]].values
    nn.fit(distances)
    distances=nn.kneighbors(distances)[0][:,1:]

    the_distance=pd.DataFrame(distances)
    the_distance = pd.DataFrame(distances)
    the_distance.columns = ["nn1", "nn2"]
    the_distance["box_num"]=len(distances)
