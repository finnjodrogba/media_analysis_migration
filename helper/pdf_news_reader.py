
import datetime
import pandas as pd
import re
import PyPDF2
import locale

class PdfNewsReader():
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
        if PdfNewsReader.__instance == None:
            PdfNewsReader()
        return PdfNewsReader.__instance

    def __init__(self):
        # Virtually private constructor
        if PdfNewsReader.__instance != None:
            raise Exception("Class PdfNewsReader is a singleton!")
        else:
            PdfNewsReader.__instance = self

    @staticmethod
    def extract_texts_to_df(pdf_path:str)->pd.DataFrame:
        """
        Extracts text between 'Body' and 'End of Document' from a PDF file,
        and returns a pandas DataFrame with the extracted texts and publication dates.
        PDF File from Lexis Nexis with some pages

        Parameters
        ----------
        pdf_path: str
            Path to the PDF file.

        Returns
        ----------
            A pandas DataFrame with columns 'Extracted Text' and 'Publication Date'.
        """

        # Function to extract all occurrences of text between "Body" and "End of Document"
        def extract_all_text_between_markers(start_marker:str, end_marker:str, reader:PyPDF2.PdfReader)->pd.DataFrame:
            #temporay storage for the document
            text = []

            # helper for date patterns
            printed = False
            loaded = False
            # Use regular expressions to find all occurrences between the markers
            pattern = re.compile(f'{re.escape(start_marker)}(.*?){re.escape(end_marker)}', re.DOTALL)
            date_pattern = re.compile(r'Load-Date: (\w+ \d{1,2}, \d{4})')

            date_pattern_zeit= re.compile("(?:\d\d|\d)\. (?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d\d\d\d")
            pattern_eod=re.compile('nd of .ocument')
            
            length_pattern=re.compile('Length: (\d+) words')
            # Basic DataFrame
            # each document will be appended to this one
            df_text = pd.DataFrame({'Extracted Text':[],'Publication Date':[],'Load Date':[],'Words':[]})
            
            for page in reader.pages:
                
                page_text = page.extract_text()
                if page_text:
                    #search for the start of the document
                    match = re.search(f'{re.escape(start_marker)}',page_text)

                
                    if match:
                        #print(match)
                        # find all dates before the start pattern
                        # there might be dates in the header line or ...
                        # but the last one is the match
                        someDates =  re.findall(date_pattern_zeit, page_text[:match.start()])
                        word_counts =  re.findall(length_pattern, page_text[:match.start()])
                        
                        if len(someDates)>0 :
                            #print (someDates)
                            # do not overwrite the day
                            if not printed:
                                #print( page_text[:match.start()] )
                                # we take the last one
                                day_of_print=datetime.datetime.strptime(someDates[-1], '%d. %B %Y')
                                printed = True
                                if  len(word_counts)>0 :
                                    word_count = word_counts[0]
                            
                    # now search for the load date        
                    load_dates=re.findall(date_pattern, page_text)
                    if len(load_dates)>0 :
                        #print (f'Load date {load_dates}')
                        # do not overwrite
                        if not loaded:
                            day_of_load=load_dates[0]
                            loaded = True


                    #store the page in the list
                    text.append(page_text)
                    
                    #skip to the end of document
                    eod=re.findall(pattern_eod, page_text)

                    
                    if len(eod) > 0 :
                        # the end of document is reached
                        #print (f'Date of Article: {day_of_print}')
                        #print (f'Date of Load: {day_of_load}')
                        #print (f'next document:{word_count}')
                        loaded = False
                        printed = False

                        #concat the pages and extract the main text
                        full_text1 = "".join(text)
                        matches = re.findall(pattern, full_text1)
                        
                        if len( matches ) > 0:
                            #print (matches[0])
                            # append the values to the DataFrame
                            df_text.loc[len(df_text)]={'Extracted Text':matches[0].strip(),'Publication Date':day_of_print,'Load Date':day_of_load,'Words':word_count}
                            # clear the document storage
                            text=[]
                        else:
                            # Shouldn't be reached
                            print('FAIL')
                    
            return df_text
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            #extracted_texts = extract_all_text_between_markers('Body', 'End of Document', reader)
            dfs = extract_all_text_between_markers('Body', 'End of Document', reader)

        # show info if necessary
        #print( dfs.info())
    
        return dfs
    
    @staticmethod
    def process_all_newspaper_articles(directory_name:str="NewsArtikel",newspaper_names = ['ZEIT', 'SPO', 'TAZ', 'WELT'],parts = [1,2,3,4,5]):
        """
        Processes all PDF files for the newspapers with a naming convention using prefixes (years 2012,2015,2023)
        and suffixes from 1 to 5.

        Parameters
        ----------
        directory_name: str
            Directory name of the files' location.

        Returns
        ----------
        pandas.DataFrame: A DataFrame containing the extracted texts and the newspaper names.
        """
        
        locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

        dataframes = []

        # Loop through the prefixes, newspaper names, and suffixes
        for prefix in [2012,2015,2023]:
            #prefix = 2012
            for newspaper_name in newspaper_names:
                for suffix in parts:
                    #suffix = 1
                    pdf_path = f'{directory_name}/{prefix}_{newspaper_name}_{suffix}.PDF'
                    print(f"Reading next file: {pdf_path}")
                    df = PdfNewsReader.extract_texts_to_df(pdf_path)  # Assuming you have a function named extract_texts_to_df
                    df['Newspaper'] = newspaper_name
                    df['Part'] = f'{prefix}_{newspaper_name}_{suffix}'

                    dataframes.append(df)

        # Concatenate all DataFrames into a single DataFrame
        final_df = pd.concat(dataframes, ignore_index=True)
        final_df['Publication Date'] = pd.to_datetime(final_df['Publication Date'])

        # Extract year from the 'Publication Date' and calculate the average SentiWS per year
        final_df['Year'] = final_df['Publication Date'].dt.year
        final_df['Words'] = final_df['Words'].astype('int64')

        return final_df


    @staticmethod
    def extract_texts_to_df_old(pdf_path):
        """
        # deprecated
        Extracts text between 'Body' and 'End of Document' from a PDF file,
        and returns a pandas DataFrame with the extracted texts and publication dates.

        :param pdf_path: Path to the PDF file.
        :return: A pandas DataFrame with columns 'Extracted Text' and 'Publication Date'.
        """

        # Function to extract all occurrences of text between "Body" and "End of Document"
        def extract_all_text_between_markers(start_marker, end_marker, reader):
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            full_text = "\n".join(text)
            
            # Use regular expressions to find all occurrences between the markers
            pattern = re.compile(f'{re.escape(start_marker)}(.*?){re.escape(end_marker)}', re.DOTALL)
            matches = re.findall(pattern, full_text)
            return [match.strip() for match in matches]

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            extracted_texts = extract_all_text_between_markers('Body', 'End of Document', reader)

        df = pd.DataFrame(extracted_texts, columns=['Extracted Text'])

        # Extract the publication date
        date_pattern = r'Load-Date: (\w+ \d{1,2}, \d{4})'
        df['Publication Date'] = df['Extracted Text'].str.extract(date_pattern)

        return df
    
    @staticmethod
    def process_all_newspaper_articles_old():
        """
        Processes all PDF files for the newspapers with a naming convention using prefixes from 1 to 3
        and suffixes from 1 to 5.

        Returns:
        pandas.DataFrame: A DataFrame containing the extracted texts and the newspaper names.
        """
        newspaper_names = ['ZEIT', 'SPO', 'TAZ', 'WELT']
        dataframes = []

        # Loop through the prefixes, newspaper names, and suffixes
        for prefix in range(1, 4):
            for newspaper_name in newspaper_names:
                for suffix in range(1, 6):
                    pdf_path = f'korpusBA/{prefix}{newspaper_name}{suffix}.PDF'
                    df = PdfNewsReader.extract_texts_to_df(pdf_path)  # Assuming you have a function named extract_texts_to_df
                    df['Newspaper'] = newspaper_name
                    dataframes.append(df)

        # Concatenate all DataFrames into a single DataFrame
        final_df = pd.concat(dataframes, ignore_index=True)
        return final_df
    
    # Define the word stems related to migration
    migration_stems = ['Migra', 'Flücht', 'Asyl', 'Einwand', 'Integrat', 'geflüchtet']

    @staticmethod
    # Function to extract sentences containing migration-related word stems
    def extract_migration_sentences(text):
        ''' Function to extract sentences containing migration-related word stems

        used words: ['Migra', 'Flücht', 'Asyl', 'Einwand', 'Integrat', 'geflüchtet']
        Returns:
        sentences which contain at least one of the word of text 
        '''
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter sentences that contain any of the migration stems
        migration_sentences = [sentence for sentence in sentences if any(stem.lower() in sentence.lower() for stem in PdfNewsReader.migration_stems)]
        # Combine filtered sentences back into a single text
        return ' '.join(migration_sentences)