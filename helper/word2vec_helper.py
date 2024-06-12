
import spacy
from gensim.models import Word2Vec
import string
import pandas as pd

nlp = spacy.load('de_core_news_sm')

def preprocess_text(text:str):
    ''' tokenize text and remove tokens of types is_stop,is_punct and not in string punctuation list

    returns: list of tokens(strings)
    '''
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.text.strip() not in string.punctuation]
    return tokens

# input for all newspapers/years: instead of specific newspaper or year(s), give 'ALL'
def find_similar_words( df:pd.DataFrame,newspaper:str, years:list|str, word:str, top_n=10):
    """ Returns similar words detected by Word2Vec

        Filtering the corpus (given as Dataframe) by newspaper and/or year

        Applies Word2Vec on the filtered corpus for the given search word

        Parameters
        ----------
        df: pd.DataFrame
            the dataframe to work with
        newspaper:str
            either a valid newspaper or 'ALL'
        years:list|str
            either a list of years [2012,2013]  or 'ALL'
        top_n : int or None, optional
            Number of top-N similar keys to return, when `topn` is int. When `topn` is None,
            then similarities for all keys are returned.

            
        Returns
        -------
        list of (str, float) or numpy.array
            When `topn` is int, a sequence of (key, similarity) is returned.
            When `topn` is None, then similarities for all keys are returned as a
            one-dimensional numpy array with the size of the vocabulary.

    """
    # filter for years and newspaper if asked for
    if newspaper != 'ALL':
        filtered_df = df[df['Newspaper'] == newspaper]
    else:
        filtered_df = df

    if years != 'ALL':
        filtered_df = filtered_df[filtered_df['Year'].isin(years)]

    # tokenize the texts
    processed_corpus = filtered_df['Extracted Text'].apply(preprocess_text).tolist()

    # train the Word2Vec-model
    model = Word2Vec(processed_corpus, vector_size=100, window=10, min_count=2, workers=4, epochs=10,  seed=42)

    # find the 10 most similar words
    similar_words = model.wv.most_similar(word, topn=top_n)
    return similar_words