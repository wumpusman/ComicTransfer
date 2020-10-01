import pandas as pd
import os
class Preprocess_Bilingual():
    """

    Attributes:
        _data_to_process: the original data in tsv format that was scraped
    """
    def __init__(self):
        """
        Preprocesses data for downstream tasks to be used for basic model analysis
        Primarily

        """
        self._data_to_process=None


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

    def set_data(self,data:pd.DataFrame):
        """

        Args:
            data: dataframe that was processed in scraping content

        Returns:
            None
        """

        self._data_to_process= self._format_data(data)

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
        return data[["font-size_jp","font-size_en"]]
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

    #https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
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
    data_path = "C://Users//egasy//Downloads//ComicTransfer//ComicTransfer//ExtractBilingual/bi/"
    data_name = "Doraemon_Long_Stories_selenium.tsv"
    full_path = os.path.join(data_path, data_name)

    all_manga = pd.read_csv(full_path, sep="\t", index_col=0)
    all_manga = all_manga.drop(columns=["level_0"])


    pre_bi=Preprocess_Bilingual()

    pre_bi.set_data(all_manga)
    data = pre_bi._data_to_process
    #test one
    assert pre_bi._data_to_process.shape == (157,40)

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

