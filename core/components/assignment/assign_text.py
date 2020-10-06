from PIL import Image, ImageFont, ImageDraw 
import numpy as np
import pandas as pd
import math
def calc_font_size(xmin:int,xmax:int,ymin:int,ymax:int, text:str)->int:
        """
        heuristic for estimating font size
        """
        width=abs(xmax-xmin)
        height=abs(ymax-ymin)

        size=int(math.sqrt((width*height)/len(text))) #How much Area Should Box Be In 

        perLine= width/size 



        return size 
    


def text_wrap(text:str, font:object, max_width:int)->list:
        """heuristic for wrapping text
        """
        lines = []
        
        # If the text width is smaller than the image width, then no need to split
        # just add it to the line list and return
        if font.getsize(text)[0]  <= max_width:
            lines.append(text)
        else:
            #split the line by spaces to get words
            words = text.split(' ')
            i = 0
            # append every word to a line while its width is shorter than the image width
            while i < len(words):
                line = ''
                while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                    line = line + words[i]+ " "
                    i += 1
                if not line:
                    line = words[i]
                    i += 1
                lines.append(line)
        return lines


class AssignDefault():
    
    def __init__(self):
        """
        the simplest most heuristic way for figuring out where text is 
        """
        self._estimated_sizes=[]

    def get_estimate_font_size(self):
        return self._estimated_sizes

    def assign_all(self,image_cv:np.array,texts:list,data:pd.DataFrame,font_path:str)->np.array:
        self._estimated_sizes = []
        image = Image.fromarray(image_cv)
        draw = ImageDraw.Draw(image)
        box_predictions=data[["x1_jp","y1_jp","x2_jp","y2_jp"]].values

        for text,box in zip(texts,box_predictions):

            xmin,ymin,xmax,ymax=box[0],box[1],box[2],box[3]
            size=calc_font_size(xmin,xmax,ymin,xmax,text)

            font = ImageFont.truetype(font_path, size)
            realigned_text = text_wrap(text, font, xmax-xmin)
            
           
            updated_font_size=calc_font_size(xmin,xmax,ymin,ymax,"\n".join(realigned_text))
         
            font = ImageFont.truetype(font_path,updated_font_size)
            self._estimated_sizes.append(updated_font_size)
            draw.text([xmin,ymin],"\n".join(realigned_text),font=font,fill=(0,0,0,255))
            
        return np.asarray(image)
    
    def assign_all1(self,image_cv:np.array,texts:list,boundings:list,font_path:str)->np.array:
        """assigns a bounding box in accordance with where the text should be """
        image= Image.fromarray(image_cv)
        draw = ImageDraw.Draw(image) 
        for bounding, text in zip(boundings,texts):
            
            aligned=pd.DataFrame(bounding)
        
            xmin=aligned["x"].min()
            xmax=aligned["x"].max()
            ymax=aligned["y"].max()
            ymin=aligned["y"].min()

            size=calc_font_size(xmin,xmax,ymin,xmax,text)
            font = ImageFont.truetype(font_path, size)
            

            realigned_text=text_wrap(text,font,xmax-xmin)
            updated_font_size=calc_font_size(xmin,xmax,ymin,ymax,"\n".join(realigned_text))
            print(updated_font_size)
            font = ImageFont.truetype(font_path,updated_font_size) 
            draw.text(aligned.values[0],"\n".join(realigned_text),font=font,fill=(0,0,0,255))
        
        return np.asarray(image)
   

        
        
        