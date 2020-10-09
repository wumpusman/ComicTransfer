from google.cloud import translate
class TranslationDefault():

    """
    Attributes:
        estimator: an object that down the line can help with reassigning text prediction given provided translations
        _should_cache: option to save current resutls
        _language: what language are you translating
        _cache: array to  temporary store ressults
    """
    def __init__(self,estimator=None,cache_results=False):
        """
        Wrapper class that just defaults for japanese to english
        """
     
        self.estimator=None
        self._should_cache:bool=cache_results #am i gonna store the original unclean data 
        self._language:str="ja" #lagnauge that this model is eexpected to be translating
        self._cache=[]#a temporary cache of previously stored results 
    
    
    def predict(self,text:str)->dict:
        raise Exception("predict not implemented")
        
    

class TranslationGoogle(TranslationDefault):
    """
    Attributes:
        client: client object from google services
        _project_id: an id identifyign a google project, required at a minimum
    """
    def __init__(self,project_id,cache_results=False):
        """
        A wrapper for google image text bounding and ocr as a first pass
        Args:
            project_id: a google api enabled project id - this project is expected to be run on the google cloud platform for specific privileges
            cache_results: do you want to save a history of your results
        """
        super().__init__(None,cache_results)
        self.client = translate.TranslationServiceClient()
        self._project_id=project_id #a requirement for translation services internally
    
    def predict(self,texts:list)->list:
        """
        return text predictions as a list
        Args:
            texts:list of japanese texts

        Returns:
            list
        """
        results=self.translate_text(texts)
        
        return [i.translated_text for i in results.translations ]
        
        
    def translate_text(self,texts:list):
        """Translating Text"""

        client = self.client

        location = "global"

        parent = f"projects/{self._project_id}/locations/{location}"
        src=self._language
        target:str="en-US"

        response = client.translate_text(
                parent= parent,
                contents= texts,
                mime_type= "text/plain",  # mime types: text/plain, text/html
                source_language_code= src,
                target_language_code= target,

        )
        return response