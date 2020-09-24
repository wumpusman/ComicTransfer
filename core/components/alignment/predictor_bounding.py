from google.cloud import vision
import io

class BoundingDefault():
    
    def __init__(self,estimator=None,cache_results=False):
        """
        essentially a default abstract class for dealing with manga
        """
        self._estimator=estimator
        self._language:str="ja" #lagnauge that this model is eexpected to be bounding
        self._should_cache:bool=cache_results #am i gonna store the original unclean data 
        self._cache=[]#a temporary cache of previously stored results 
        print("HUH")
    def set_language(self,name:str)->None:
        self._language=name
    
    def get_language(self)->str:
        return self._language
    
    def predict(self,path:str)->dict:
        
        raise Exception("predict not implemented")
    
class BoundingGoogle(BoundingDefault):
    
    def __init__(self,cache_results=False):
        """
        A wrapper for google image text bounding and ocr as a first pass
        """
        super().__init__(None,cache_results)
        self.client=vision.ImageAnnotatorClient()
        
       
    def predict(self,path:str)->dict:
        """
        get bounding box predictions
        """
        text,bounding,response= self._detect_document(path) 
        print(self._cache)
        if self._should_cache:
            print("GREAT")
            self._cache.append([{"path":path,"response":response}])
        
      
        return {"text":text,"vertices":bounding}
    
    
    
    def _detect_document(self,path):
        """Detects document features in an image."""

        client = vision.ImageAnnotatorClient()

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.types.Image(content=content)

        response = client.document_text_detection(image=image)


        all_words=[]
        bounding_boxes=[]
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
               # print('\nBlock confidence: {}\n'.format(block.confidence))

                for paragraph in block.paragraphs:
                   # print('Paragraph confidence: {}'.format(
                    #    paragraph.confidence))

                    xs=[]
                    ys=[]
                    for vertice in paragraph.bounding_box.vertices:
                        xs.append(vertice.x)
                        ys.append(vertice.y)

                    words=""


                    for word in paragraph.words:
                        word_text = ''.join([
                            symbol.text for symbol in word.symbols
                        ])
                       # print('Word text: {} (confidence: {})'.format(
                          #  word_text, word.confidence))

                        words+=word_text
                        #for symbol in word.symbols:
                           # print('\tSymbol: {} (confidence: {})'.format(
                              #  symbol.text, symbol.confidence))
                    all_words.append(words)
                    bounding_boxes.append({"x":xs,"y":ys})
        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))
        return all_words,bounding_boxes,response
    
    def _detect_text(self,path:str, lang="ja"):
        """Detects text in the file
        """
    
        client = self.client

        with io.open(path, 'rb') as image_file:
            content = image_file.read()
        

        image = vision.types.Image(content=content)

        response = client.text_detection(image=image,image_context={"language_hints": [lang]})
        texts = response.text_annotations
        ''' 
        print('Texts:')

        for text in texts:
            print('\n"{}"'.format(text.description))

            vertices = (['({},{})'.format(vertex.x, vertex.y)
                        for vertex in text.bounding_poly.vertices])

            print('bounds: {}'.format(','.join(vertices)))

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))
        '''
        return response
    
    def _format_to_dictionary_sentence(self,google_response)->dict:
        pass
    
    def _format_to_dictionary_words(self,google_response)->dict:
        """
        Converts google response object to a dicitonary with relevant values for analysis
        """
        to_dict={}
        
        to_dict["locale"]=google_response[0].locale
        to_dict["ocr"]=google_response[0].description
        to_dict["texts"]=[]
        to_dict["vertice"]=[]
        
        for annotation in google_response[1:]:
            to_dict["texts"].append(annotation.description)
            to_dict["vertice"].append({"x":[],"y":[]})
            for vertice in annotation.bounding_poly.vertices:
                coord_dict=to_dict["vertice"][-1]

                coord_dict["x"].append(vertice.x)
                coord_dict["y"].append(vertice.y)
    
        return to_dict


