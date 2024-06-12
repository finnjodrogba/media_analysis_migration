import spacy
import pandas as pd
from spacy_sentiws import spaCySentiWS
from spacy.language import Language


class SentiWS_Metric():
    """ class to calculate Sentiment WS

    by splitting text into sentences and calculating the value per sentence.
    Splitting is done by using nlp - spaCy
    The word score (sentiws) is directly calculated by adding sentiws to the pipe and then
    automatically added to the word.
    There is one base function (analyze_sentiment_ws_text),  just adding up the values per sentence
    returns polarity (pos and neg) and weighted polarity (pos and neg)
    Works as a singleton, initialisation done by init function of singleton
    """
    __instance = None
    

    @staticmethod
    def getInstance():
        # Static access method. 
        if SentiWS_Metric.__instance == None:
            SentiWS_Metric()
        return SentiWS_Metric.__instance

    def __init__(self):
        # Virtually private constructor
        if SentiWS_Metric.__instance != None:
            raise Exception("Class SentiWS_Metric is a singleton!")
        else:
            SentiWS_Metric.__instance = self

        # load spacy for german
        self.nlp = spacy.load('de_core_news_sm')

        # loads the sentiment ws data (old approach)
        #self.sentiws = spaCySentiWS("data/sentiws")
        #self.nlp.add_pipe(self.sentiws)
        # new spaCy 3.0 approach with sentiws added
        self.nlp.add_pipe('sentiws', config={'sentiws_path': 'data/sentiws/'})
        self.used_words={}

    def analyze_sentiment_ws_tokens(self,token_list):
        """Return a list of polarity values

        Takes the mean of  all values of the word's sentiws value (other than zero)
        If no word with a sentiws value is found, the return value will be [0,0,0,0]

        Parameters
        ----------
        token_list : list
            list of tokens - given by tokenizing with spacy

        Returns
        -------
        list
            a counter of positive senitws values found in the token list

            a counter of negative senitws values 

            a sum of  positive senitws values 

            a sum of  negative senitws values 
         """
        if not token_list:
            return [0,0,0,0]

        if token_list=="":
            return [0,0,0,0]

        
        # Remove tokens not required
        words=[word for word in token_list if word.pos_ not in ["SPACE","PUNCT"]]

       
        try:
            
            #initializing return values
            pos=0
            negs=0
            pos_values=0
            neg_values=0
            #used_words={}

            #loop through the words
            for word in words:
                
                #if it has a sentiws value we add the results accordingly
                if word._.sentiws is not None:
                    #print(f'Found word: {word._.sentiws} {word}')
                    self.used_words[word.text]= word._.sentiws
                    if word._.sentiws > 0:
                        pos +=1
                        pos_values += word._.sentiws
                    else:
                        negs +=1
                        neg_values += word._.sentiws
                #else:
                #    print(f'nada:  {word}')
                #i += 1

            return [pos,negs,pos_values,neg_values]
            
        except Exception as e:
            print(type(e))    # the exception type
            print(e.args)     # arguments stored in .args
            print(e) 
            print( f'Error :{token_list}' )
            return [0,0,0,0]

    @staticmethod
    def analyze_sentiment_ws_text(text:str):
        """Calculates the polarity and the polarity numbers of the text.

        Parameters
        ----------
        text : str
            The text to analyze
        """
        if pd.isna( text):
            return [0,0,0,0]

        score_doc= [0,0,0,0]
       
        sentiws_3 = SentiWS_Metric.getInstance()

        sentiws_3.used_words={}

        doc=sentiws_3.nlp(text)

        result = sentiws_3.analyze_sentiment_ws_tokens(doc)
        for i in range(len(result)):
            score_doc[i] += result[i]

        return score_doc

    @staticmethod
    def analyze_sentiment_ws_text_sentence(text):
        
        """Calculates the polarity and the polarity numbers of the text.
        
        Split into sentences, call per sentence 

        Parameters
        ----------
        text : str
            The text to analyze
        """
        #skip empty entries
        if pd.isna( text):
            return [0,0,0,0]

        score_doc= [0,0,0,0]
       

        sentiws_3 = SentiWS_Metric.getInstance()

        sentiws_3.used_words={}
        
        #split textt into document
        doc=sentiws_3.nlp(text)
        for sentence in doc.sents:
            #add results
            result = sentiws_3.analyze_sentiment_ws_tokens(sentence)
            for i in range(len(result)):
                score_doc[i] += result[i]

            # and append list of words
            # used_words=used_words | sentiws_3.used_words

        #print(used_words) 
        return score_doc
    
    @staticmethod
    def get_used_words() -> dict:
        """ Gets the used words in last  """
        return SentiWS_Metric.getInstance().used_words
    
    @staticmethod
    def get_fail_text() -> str:
        """ Gets an example string that will not be calculated correctly """
        fail_text='''Sie flüchten und viele sterben - doch die Boote kommen weiterhin'''
        
        return fail_text
    
    @staticmethod
    def fail_example():
        """ Prints an example string that will not be calculated correctly """
        fail_text = SentiWS_Metric.get_fail_text()
        print( fail_text)
        print( 'Result:',SentiWS_Metric.analyze_sentiment_ws_text(fail_text))
        print('flüchten not in negative words','Sterben negative only as noun, not as verb')

if __name__ == "__main__":
    #df=pd.read_csv('example.csv')
    
    SentiWS_Metric.fail_example()

    text ='''Nassau - Ein mit geschätzt 150 haitianischen Flüchtlingen vollkommen überladenes Segelboot 
    ist vor den Bahamas gekentert und hat mindestens 30 Menschen in den Tod gerissen. 
    Stundenlang hätten sich die Überlebenden an den Rand des 13 Meter langen Bootes geklammert, 
    bis Hilfe kam, sagte ein Behördensprecher. Die genaue Zahl der Toten sei derzeit nicht bekannt. 
    Bestätigt sind 20 Opfer, laut Berichten von Flüchtlingen an Bord sollen es mindestens 30 sein. 
    Militär und Polizei auf den Bahamas arbeiten eng mit der US-Küstenwache zusammen, um die Leichen zu bergen. 
    110 Menschen seien gerettet worden, sagte ein Regierungssprecher in Nassau dem "Miami Herald". 
    Die Flüchtlinge sollen acht oder neun Tag auf See unterwegs gewesen sein - mit wenig Trinkwasser und Lebensmitteln, außerdem ohne Rettungswesten. 
    Viele Überlebende seien extrem dehydriert gewesen, als die Rettungsmannschaften sie fanden.
      "Das Boot war eindeutig völlig überladen, instabil und nicht seetüchtig", sagte Leutnant Gabe Somma von der Küstenwache. Zahlreiche Haitianer versuchen jeden Monat, 
      illegal in die Vereinigten Staaten zu gelangen. 
      Auch die östlich von Haiti liegende, zu den USA gehörende Karibikinsel Puerto Rico hat in letzter Zeit verstärkt haitianische Flüchtlinge vor ihrer Küste aufgegriffen. 
      ala/dpa/AP Load-Date: November 27, 2013'''

    print (SentiWS_Metric.analyze_sentiment_ws_text(text))

    print( SentiWS_Metric.get_used_words() )

    print (SentiWS_Metric.analyze_sentiment_ws_text_sentence(text))

