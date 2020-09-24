from google.cloud import translate
class TranslationDefault():
    
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
    
    def __init__(self,project_id,cache_results=False):
        """
        A wrapper for google image text bounding and ocr as a first pass
   
        """
        super().__init__(None,cache_results)
        self.client= client = translate.TranslationServiceClient()
        self._project_id=project_id #a requirement for translation services internally
    
    def predict(self,texts:list)->list:
        
        results=self.translate_text(texts)
        
        return [i.translated_text for i in results.translations ]
        
        
    def translate_text(self,texts:list):
        """Translating Text."""

        client = self.client

        location = "global"

        parent = f"projects/{self._project_id}/locations/{location}"
        src=self._language
        target:str="en-US"
        # Detail on supported types can be found here:
        # https://cloud.google.com/translate/docs/supported-formats
        response = client.translate_text(
                parent= parent,
                contents= texts,
                mime_type= "text/plain",  # mime types: text/plain, text/html
                source_language_code= src,
                target_language_code= target,

        )
        return response