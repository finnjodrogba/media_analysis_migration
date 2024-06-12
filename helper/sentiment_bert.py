
import pandas as pd
import time
import os
import platform
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import json

class SentimentBert():
    """ class to calculate Sentiment via Bert

    
    """
    __instance = None
    

    @staticmethod
    def getInstance():
        # Static access method. 
        if SentimentBert.__instance == None:
            SentimentBert()
        return SentimentBert.__instance

    def __init__(self):
        # Virtually private constructor
        if SentimentBert.__instance != None:
            raise Exception("Class SentimentBert is a singleton!")
        else:
            SentimentBert.__instance = self



    model_name = "mdraw/german-news-sentiment-bert"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,truncation = True)

    @staticmethod
    def calculate_sentiment( df:pd.DataFrame, text_column:str, result_column:str, index_file_name ,tmp_file_name='df_korpus_tmp.csv',modulus:int=100, sleeptime_in_sec:int=20, save_df:bool=True):
        """ calculates the sentiment Bert into a new column of the data frame

        After some amount of records the function will sleep for a while to prevent os crashes - happened on a MacBook

        To store the state, an index file will store  the last calculated row index. If this file exists and contains a number, calculating will start at that given index

        Also the dataframe will be stored in a csv file (depending on save_df)

        Parameters
        ----------
        df: pd.DataFrame
            the dataframe to work with
        text_column:str
            the text column of the dataframe that should be analyzed
        result_column:str
            the column where the result will be stored
        index_file_name:str
            the name of the index file
        tmp_file_name:str
            Resulting dataframe will be stored in this file
        modulus: int
            the amount of rows/texts that will be calculated before sleeping
        sleeptime_in_sec:int
            time in seconds 
        save_df: bool
            if temporary results should be store
            
        """
        try:
            # Read the last index if the file exists
            if os.path.exists(index_file_name):
                with open(index_file_name, "r") as file:
                    start_index = int(file.read().strip())
            else:
                start_index = 0
        
            print(f"Starting at index {start_index}.")
            
            for i in range(start_index, len(df)):
                #print(f"running: {i} of {len(df)}")
                if not pd.isnull(df.at[i, text_column]):
                    df.at[i, result_column] = SentimentBert.sentiment_pipeline(df.at[i, text_column])
                    if i> start_index and i % modulus == (start_index % modulus):
                        if save_df:
                            df.to_csv('df_korpus_tmp.csv', index=False)
                            with open(index_file_name, "w") as file:
                                file.write(str(i))
                        
                        print(f"Sleep {sleeptime_in_sec} seconds at index {i}")
                        time.sleep(sleeptime_in_sec)
                        print(f"Continuing")
                
            df.to_csv(tmp_file_name, index=False)
            print("Finished, clean up")  

            try:
                os.remove(index_file_name) 
            except:
                print('Finished')
        
        except KeyboardInterrupt:
            # Save the current index on interruption
            with open(index_file_name, "w") as file:
                file.write(str(i))
            df.to_csv(tmp_file_name, index=False)
            print(f"Interrupted by user. Progress saved up to index {i}.")

    def calculate_sentiment_nobreak( df:pd.DataFrame, text_column:str, result_column:str,tmp_file_name:str):
        """
        calculates the sentiment Bert into a new column of the data frame

        To store the state, an index file will store  the last calculated row index 

        Also the dataframe will be stored in a csv file (depending on save_df)

        Parameters
        ----------
        df: pd.DataFrame
            the dataframe to work with
        text_column:str
            the text column of the dataframe that should be analyzed
        result_column:str
            the column where the result will be stored
        tmp_file_name:str
            Resulting dataframe will be stored in this file
            
        """
        try:
            # Read the last index if the file exists
            
            start_index = df.index.min()
            end_index = df.index.max()
            print(f"Starting at index {start_index}.")
            
            for i in range(start_index,end_index+1):
                if i%100 == 0:
                    print(f"running: {i} of {start_index} {end_index + 1}")
                if not pd.isnull(df.at[i, text_column]):
                    df.at[i, result_column] = SentimentBert.sentiment_pipeline(df.at[i, text_column])
                    
            print("storing file")  

            df.to_csv(tmp_file_name, index=False)
    
        
            print("Finished")  
        
        except KeyboardInterrupt:
            # Save the current index on interruption
            #with open(index_file_name, "w") as file:
            #    file.write(str(i))
            #df.to_csv(tmp_file_name, index=False)
            print(f"Interrupted by user. Progress saved up to index {i}.")
            
    def sentiment_to_score(sentiment):
        """
        Returns the score of the sentiment given in the form
        "[{'label': 'neutral', 'score': 0.5374473929405212}]"

        if label is 'neutral', the return value will be 0
        Otherwise the score returned will be the score value (multiplied by -1 if label is 'negative')

            
        Returns
        -------
        float
            the score as float

        """
        
        #print( type(sentiment))
        # handle string input or list input 
        if isinstance(sentiment, str) :
            # strings must be converted to a sentiment object
            my_list = json.loads( sentiment.replace("'","\"") )
            sentiment_dict = my_list[0]  
            #print( 'str',my_list[0] )
        else:
            # Access the first dictionary in the list
            # sentiment_dict = sentiment[0]
            #print( 'obj',type(sentiment) )
            #print(sentiment[0])
            sentiment_dict = sentiment[0]

        
        label = sentiment_dict['label']
        score = sentiment_dict['score']
        if label == 'negative':
            return -1 * score  # Negative scores as negative values
        elif label == 'positive':
            return score  # Positive scores as positive values
        else:  # Neutral
            return 0
          
    #model requires text of max length of 512 words
    def truncate_text(text):
        return text[:512]
    
    @staticmethod
    def truncate_text2(text):
        # Split the text into words and truncate to the first 512 words
        truncated_words = text.split()[:512]
        # Join the words back into a string
        return ' '.join(truncated_words)
    
    @staticmethod
    def analyze_sentiment(text):
        truncated_text = SentimentBert.truncate_text(text)
        result = SentimentBert.sentiment_pipeline(truncated_text)
        return result
    
    #results of Sentiment Analysis are in form: {label: x, score: y}; convert them to one score
    @staticmethod
    def sentiment_to_score_old(sentiment):
        # Access the first dictionary in the list
        sentiment_dict = sentiment[0]
        label = sentiment_dict['label']
        score = sentiment_dict['score']
        if label == 'negative':
            return -1 * score  # Negative scores as negative values
        elif label == 'positive':
            return score  # Positive scores as positive values
        else:  # Neutral
            return 0
        

"""
if platform.platform(terse=True)[:5] == "macOS":
    print( 'OS platform:', platform.platform() )
    SentimentBert.calculate_sentiment(df,text_column='Extracted Text',result_column='Sentiment',index_file_name='last_index.txt',tmp_file_name='df_korpus_tmp.csv',modulus=100,sleeptime_in_sec=20)
    SentimentBert.calculate_sentiment(df,text_column='MigText',result_column='Sentiment_MigText',index_file_name='last_index.txt',tmp_file_name='df_korpus_tmp.csv',modulus=100,sleeptime_in_sec=20)

else:
    print( 'Other platform:', platform.platform() )

    SentimentBert.calculate_sentiment_nobreak(df,text_column='Extracted Text',result_column='Sentiment',tmp_file_name='df_korpus_tmp.csv')
    SentimentBert.calculate_sentiment_nobreak(df,text_column='MigText',result_column='Sentiment_MigText',tmp_file_name='df_korpus_tmp.csv')

df['Sentiment_Score'] = df['Sentiment'].apply(SentimentBert.sentiment_to_score)
df['SentiScore_Migtext'] = df['Sentiment_MigText'].apply(SentimentBert.sentiment_to_score)
"""
