import nltk
import spacy
from collections import Counter

class Noun_Counter():
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
        if Noun_Counter.__instance == None:
            Noun_Counter()
        return Noun_Counter.__instance

    def __init__(self):
        if Noun_Counter.__instance != None:
            raise Exception("Class SentiWS_Metric is a singleton!")
        else:
            Noun_Counter.__instance = self


    def set_stopwords( self, stopwords ):
        """
        # sets stopwords which may be already loaded
        otherwise loads the stopwords using nltk
        loaded with nltk.download('stopwords'), migth be modified with own stop-words like union
        german_stopwords = set(nltk.corpus.stopwords.words('german'))
        and add some words
        custom_stopwords = german_stopwords.union({"load-date", "page"})    

        Parameters
        ----------
        stopwords:
            stopwords
        
        """
        if stopwords == None :
            nltk.download('stopwords') 
            self.stopwords = set(nltk.corpus.stopwords.words('german'))
            # Füge benutzerdefinierte Stopwörter hinzu 
            #self.stopwords = german_stopwords.union({"load-date", "page"})
        else:
            self.stopwords = stopwords

    def add_custom_words(self,words):
         self.stopwords = self.stopwords.union(words)

    def set_nlp(self,nlp):
        """
        # set nlp or load via spacy if nlp is none

        so it occuld be sett if already loaded
        nlp = spacy.load('de_core_news_sm')

         Parameters
        ----------
        nlp:
            nlp loaded by spacy

        """
        if nlp == None :
            self.nlp = spacy.load('de_core_news_sm')
        else:
            self.nlp=nlp


    def get_most_common_nouns(self,texts, top_n=20): 
        all_nouns = [] 
        for doc in self.nlp.pipe(texts, disable=["parser", "ner"]): 
            nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN" and token.text.lower() not in self.stopwords] 
            all_nouns.extend(nouns) 
        return Counter(all_nouns).most_common(top_n)
    
    