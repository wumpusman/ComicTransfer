import pandas as pd
import os

from sklearn.neighbors import NearestNeighbors
import numpy as np

def read_tsv_files(paths:list)->pd.DataFrame:
    """
      aggregates multiple selenium tsv results into a single large file
    Args:
        paths: a list of paths to associated tsv results

    Returns:
        pd.Dataframe
    """

    all_mangas_pds:list=[]
    for path in paths:
        full_path = path

        all_manga = pd.read_csv(full_path, sep="\t", index_col=0)
        if len(all_manga)>0:
            all_manga = all_manga.drop(columns=["level_0"])
        all_mangas_pds.append(all_manga)

    all_manga = pd.concat(all_mangas_pds,sort=False)

    return all_manga

class Preprocess_Bilingual():
    """

    Attributes:
        _data_to_process: the original data in tsv format that was scraped
    """
    def __init__(self):
        """
        Preprocesses data for downstream tasks to be used for basic model analysis
        asssociated with extraction done by extract_img_selenium

        """
        self._data_to_process=None

    @staticmethod
    def convert_features_from_raw_page(page_data:pd.DataFrame)->pd.DataFrame:
        """
        converts features from a raw page into a similar form of features in bilingual process
        Args:
            page_data: a dataframe that has data with the columns
            ['x1_jp','y1_jp','x2_jp','y2_jp','top_jp','left_jp','width_jp',
                   'height_jp','text_jp_len',"text_jp"]

        Returns:
            pd.DataFrame
        """

        expected_minimum_feature_names = ['width_jp', 'height_jp', 'top_jp', 'left_jp',
                           'x1_jp', 'y1_jp', 'x2_jp', 'y2_jp', 'text_jp_len',
                           'nn1', 'nn2', 'box_num', 'x1_jp_squared',
                           'y1_jp_squared', 'x2_jp_squared', 'y2_jp_squared',
                           'width_jp_squared']

        Preprocess_Bilingual._add_meta_information(page_data)

        squared=page_data[['x1_jp', 'y1_jp', 'x2_jp', 'y2_jp']]
        squared=squared*squared
        squared=pd.DataFrame(squared.values)
        squared.columns=['x1_jp_squared','y1_jp_squared', 'x2_jp_squared', 'y2_jp_squared']


        jp_area=page_data[['width_jp', 'height_jp']]
        jp_area=jp_area*jp_area
        jp_area=pd.DataFrame(jp_area)
        jp_area.columns=["width_jp_squared","height_jp_squared"]

        return pd.concat([page_data,squared,jp_area],axis=1)



    def output_all_features_font_size(self,normalize:bool=False)->tuple:
        """
        outputs all features relevant to font size as a tuple of dataframes X, y
        Args:
            normalize: return certain features normalized by page size

        Returns:
            tuple
        """
        x_feature_names = ['width_jp', 'height_jp', 'top_jp', 'left_jp',
                           'x1_jp', 'y1_jp', 'x2_jp', 'y2_jp', 'text_jp_len',
                           'nn1', 'nn2', 'box_num', 'x1_jp_squared',
                           'y1_jp_squared', 'x2_jp_squared', 'y2_jp_squared',
                           'width_jp_squared']

        y_feature_names = ['font-size_en']

        all_pd = self.output_all_features(normalize)
        x_pd = all_pd[x_feature_names]
        y_pd = all_pd[y_feature_names]
        return x_pd, y_pd

    def output_all_features_iou(self,normalize:bool=False)->tuple:
        """
        outputs all features relevant to bounding box  as a tuple of dataframes X, y
        Args:
            normalize: return certain features normalized by page size

        Returns:
            tuple
        """
        x_feature_names = ['width_jp', 'height_jp', 'top_jp', 'left_jp',
                           'x1_jp', 'y1_jp', 'x2_jp', 'y2_jp', 'text_jp_len',
                           'nn1', 'nn2', 'box_num', 'x1_jp_squared',
                           'y1_jp_squared', 'x2_jp_squared', 'y2_jp_squared',
                           'width_jp_squared']

        y_feature_names = ['x1_en', 'y1_en', 'x2_en', 'y2_en']

        all_pd=self.output_all_features(normalize)
        x_pd=all_pd[x_feature_names]
        y_pd=all_pd[y_feature_names]
        return x_pd,y_pd

    def output_all_features(self, normalize:bool=False)->pd.DataFrame:
        """
        aggregates the organized features in a matrix
        Args:
            normalize: are input features normalized by size of page if relevant

        Returns:
            pd.DataFrame
        """

        X_feature_names = ['width_jp', 'height_jp', 'top_jp', 'left_jp',
                           'x1_jp', 'y1_jp', 'x2_jp', 'y2_jp', 'text_jp_len',
                            'nn1', 'nn2', 'box_num', 'x1_jp_squared',
                           'y1_jp_squared', 'x2_jp_squared', 'y2_jp_squared',
                           'width_jp_squared','font-size_jp']

        y_feature_names = ['x1_en', 'y1_en', 'x2_en', 'y2_en']

        box_area = self.extract_box_area(normalize)
        box_location = self.extract_box_location(normalize)
        box_coords = self.to_box_coords(normalize)
        text_info = self.extract_text_length()
        font_size = self.extract_font_size()
        neighbors_and_boxes_on_page=self.extract_macro_features()
        squared_features=self.extract_squared_features(normalize)

        pd_features=self.aggregate_to_pandas([box_area,box_location,box_coords,text_info,font_size,neighbors_and_boxes_on_page,squared_features])

        return pd_features

    def aggregate_to_pandas(self,processed_data:list)->pd.DataFrame:
        """

        Args:
            processed_data: a list of dictionaries of data

        Returns:
            pd.DataFrame
        """

        pd_form:list=[]

        for i in processed_data:
            pd_form.append(pd.DataFrame(i))

        return pd.concat(pd_form,axis=1)

    @staticmethod
    def _add_meta_information(original_data_frame):
        """gives information about nearest neighbor distance, and box info on the page
           to data being processed
        Returns:
            None
        """
        if ("link" in original_data_frame.columns)==False: #in case there is no link such as when predicting
            original_data_frame["link"]="empty"

        coords_names = ["x1_jp", "y1_jp", "x2_jp", "y2_jp"]

        unique_pages = original_data_frame.link.unique()
        links = original_data_frame.groupby("link")
        all_meta = []
        for page in unique_pages:
            specific_page = links.get_group(page)
            temp_process = Preprocess_Bilingual()
            temp_process._data_to_process = specific_page.reset_index()

            #check if coords already exist

            coords = None
            coords_exist =  all(elem in original_data_frame.columns for elem in coords_names)
            if coords_exist==False:
                coords = pd.DataFrame(temp_process.to_box_coords(False))
            else:
                coords =original_data_frame[coords_names]

            if len(coords) >= 3:
                nn = NearestNeighbors(n_neighbors=3)
                distances = coords[coords_names].values

                nn.fit(distances)
                distances = nn.kneighbors(distances)[0][:, 1:]
                the_distance = pd.DataFrame(distances)
                the_distance.columns = ["nn1", "nn2"]
                the_distance["box_num"] = len(distances)
                all_meta.append(the_distance)
            else:
                the_distance = pd.DataFrame(np.zeros((len(coords), 2)))
                the_distance.columns = ["nn1", "nn2"]
                the_distance["box_num"] = len(coords)
                all_meta.append(the_distance)

        additional_features=pd.concat(all_meta)
        original_data_frame["nn1"]=additional_features["nn1"].values
        original_data_frame["nn2"]=additional_features["nn2"].values
        original_data_frame["box_num"]=additional_features["box_num"].values

    def set_data(self,data:pd.DataFrame):
        """
            Preps and aligns data that was scraped into even groups, and adds feature extraction
        Args:
            data: dataframe that was processed in scraping content

        Returns:
            None
        """

        self._data_to_process= self._format_data(data)
        self._add_meta_information(self._data_to_process)

    def _format_data(self,manga_pd:pd.DataFrame)->pd.DataFrame:
        """aligns and formatting japanese and english meta features from scraped data that is used for downstream analysis

        Args:
            manga_pd: a dataframe consisting of a prespecified format with various features

        Returns:

            pd.Dataframe: aligned data
        """

        japanese = manga_pd[manga_pd.language == "jp"]
        english = manga_pd[manga_pd.language == "en"]
        aligned_data=pd.merge(japanese,english,how="inner",on=["manga","link","boxID","id"],suffixes=("_jp","_en"))
        return aligned_data

    def extract_text(self)->dict:
        """
            extracts translations for both languages side by side,
        Returns:
            dict
        """

        data=self._data_to_process
        return data[["text_jp","text_en"]].to_dict()

    def extract_text_length(self)->dict:
        """
            extracts the text length of both languages
        Returns:
            dict
        """
        length_pd=pd.DataFrame()
        data = self._data_to_process
        length_pd["text_jp_len"] = data["text_jp"].str.len().values
        length_pd["text_en_len"] = data["text_en"].str.len().values

        return length_pd.to_dict()

    def extract_fonts(self)->dict:
        """
            extracts fonts for both languages side by side
        Returns:
            dict
        """
        data = self._data_to_process
        return data[["font-family_jp", "font-family_en"]].to_dict()

    def extract_font_size(self)->dict:
        """
            extracts fonts for both languages side by side

        Returns:
            dict
        """
        data = self._data_to_process
        return data[["font-size_jp","font-size_en"]].to_dict()


    def extract_box_location(self,normalize:bool=True)->dict:
        """
            returns 4 box pairs normalized or unnormalized relative to their size [x,y,x1,y1]
        Args:
            normalize: normalize the box locations relative to the image size

        Returns:
            dict
        """
        data = self._data_to_process
        page_height: int = data["page_height_jp"]
        page_width: int = data["page_width_jp"]

        l_jp = data["left_jp"]
        t_jp = data["top_jp"]
        l_en = data["left_en"]
        t_en = data["top_en"]
        new: pd.DataFrame = pd.DataFrame()
        new["top_jp"] = t_jp
        new["left_jp"] = l_jp
        new["left_en"] = l_en
        new["top_en"] = t_en

        if normalize:
            new["left_jp"] = new["left_jp"] / page_width
            new["left_en"] = new["left_en"] / page_width
            new["top_en"] = new["top_en"] / page_height
            new["top_jp"] = new["top_jp"] / page_height

        return new.to_dict()

    def _to_box_coords(self,lang="jp", normalize: bool = True)->pd.DataFrame:

        """
            takes in size and offset and makes into x1,x2,y1,y2 format (top left, bottom right)
        Args:
            lang: language to select either ["jp", "en"]
            normalize: is this absolute or relative

        Returns:
            pd.DataFrame
        """
        box_area: dict = self.extract_box_area(normalize)
        box_location: dict = self.extract_box_location(normalize)

        area = pd.DataFrame(box_area)
        location = pd.DataFrame(box_location)

        lang_area = area[["width_{}".format(lang), "height_{}".format(lang)]]
        lang_location = location[["left_{}".format(lang), "top_{}".format(lang)]]

        bottom_right = lang_area.values + lang_location.values
        bottom_right_pd = pd.DataFrame(bottom_right)
        bottom_right_pd.columns = ["x2_{}".format(lang), "y2_{}".format(lang)]

        top_left = pd.DataFrame(lang_location)
        top_left.columns = ["x1_{}".format(lang), "y1_{}".format(lang)]

        final_format = pd.concat([top_left, bottom_right_pd], axis=1)

        return final_format




    def to_box_coords(self,normalize:bool=True)->dict:
        """
            takes in size and offset of jp and english and returns
             x1,x2,y1,y2 format (top left, bottom right) for jp and en
        Args:
            normalize: is this assuming box of 1 or actual image size

        Returns:
            dict
        """

        jp_data:pd.DataFrame=self._to_box_coords("jp",normalize)
        en_data:pd.DataFrame=self._to_box_coords("en",normalize)

        return pd.concat([jp_data,en_data],axis=1).to_dict()

    def extract_squared_features(self, normalize:bool=True)->dict:


        jp_area=pd.DataFrame(self.extract_box_area(normalize))[["width_jp","height_jp"]]
        jp_area=jp_area*jp_area
        jp_area.columns=["width_jp_squared","height_jp_squared"]

        jp_loc=pd.DataFrame(self.to_box_coords(normalize))[["x1_jp","y1_jp","x2_jp","y2_jp"]]
        jp_loc=jp_loc*jp_loc
        jp_loc.columns=["x1_jp_squared","y1_jp_squared","x2_jp_squared","y2_jp_squared"]

        return pd.concat([jp_loc,jp_area],axis=1).to_dict()



    def extract_macro_features(self)->dict:
        """
            get macro features associated with the jp box info
        Returns:
            dict
        """
        return self._data_to_process[["nn1","nn2","box_num"]].to_dict()

    def extract_box_area(self,normalize:bool=True)->dict:
        """
            returns box area normalized or unnormalized relative to their size [w,h,w1,h1]
        Args:
            normalize: normalize the box locations relative to the image size

        Returns:
            dict
        """
        data = self._data_to_process
        page_height: int = data["page_height_jp"]
        page_width: int = data["page_width_jp"]

        w_jp = data["innerWidth_jp"]
        h_jp = data["innerHeight_jp"]
        w_en = data["innerWidth_en"]
        h_en = data["innerHeight_en"]
        new: pd.DataFrame = pd.DataFrame()
        new["width_jp"] = w_jp
        new["height_jp"] = h_jp
        new["width_en"] = w_en
        new["height_en"] = h_en



        if normalize:
            new["width_jp"] = new["width_jp"] / page_width
            new["width_en"] = new["width_en"] / page_width
            new["height_en"] = new["height_en"] / page_height
            new["height_jp"] = new["height_jp"] / page_height


        return new.to_dict()



if __name__ == '__main__':





    data_path = "/home/jupyter/ComicTransfer2/ComicTransfer/data/bilingual_tsv"
    data_name = "Doraemon_Long_Stories_selenium.tsv"
    full_path = os.path.join(data_path, data_name)

    all_manga = pd.read_csv(full_path, sep="\t", index_col=0)
    all_manga = all_manga.drop(columns=["level_0"])


    pre_bi=Preprocess_Bilingual()

    pre_bi.set_data(all_manga)
    data = pre_bi._data_to_process
    #test one
    assert pre_bi._data_to_process.shape == (157,43)

    #test two
    temp=pre_bi.extract_text()
    assert temp['text_en'][0] == "AH, THAT'S AWESOME!"
    print("OK")

    #test three
    temp=pre_bi.extract_text_length()
    assert temp["text_jp_len"][0]==8
    assert temp["text_en_len"][1]==13
    print("OK")

    #test four
    temp=pre_bi.extract_fonts()
    assert temp["font-family_jp"][0]==' yasashisa-antique, sans-serif'

    #test five
    temp = pre_bi.extract_font_size()
    assert temp["font-size_jp"][0] == 20

    # test six
    temp = pre_bi.extract_box_area(normalize=False)
    assert temp["width_jp"][0] == 64
    temp = pre_bi.extract_box_area(normalize=True)
    assert temp["width_jp"][0] == .08

    #test seven
    temp = pre_bi.extract_box_location(normalize=True)
    assert temp["left_jp"][4]==.04250
    temp = pre_bi.extract_box_location(normalize=False)
    assert temp["left_jp"][4] == 34

    #test eight
    temp1=pre_bi.extract_box_location(normalize=True)
    temp2=pre_bi.extract_font_size()
    temp3=pre_bi.extract_text_length()
    joined_temp=pre_bi.aggregate_to_pandas([temp1,temp2,temp3])
    assert joined_temp.shape == (157,8)

    #get the right bounding boxes
    temp = pre_bi.to_box_coords(False)
    assert (int(temp["x2_jp"][2]),int(temp["y2_jp"][2]))==(350,1069)


    temp = pre_bi.extract_macro_features()

    assert (int(temp["nn1"][0])==309)
    assert (int(temp["nn2"][0]) == 667)
    assert (int(temp["box_num"][0])==6)


    temp=pd.DataFrame((pre_bi.extract_squared_features(False)))
    assert(int(temp["x1_jp_squared"][0])==(410881))
    assert(int(temp["height_jp_squared"][1]==(5184)))


    temp=pre_bi.output_all_features()
    x, y = pre_bi.output_all_features_font_size()
    assert (x.shape==(157,17))
    assert (y.shape)==(157,1)

    x, y = pre_bi.output_all_features_iou()
    assert (x.shape==(157,17))
    assert (y.shape)==(157,4)
    print("OK")
