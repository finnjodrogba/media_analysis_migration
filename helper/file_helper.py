import pandas as pd

def read_file(filename:str = 'korpus_calculated.csv', cleanup:bool=True,show_info:bool=True,empty_dataframe:bool=True)->pd.DataFrame:
    ''' Reads a previousöy saved csv file into a dataframe

    Parameters
    ----------
    filename : str
        the name and location of the file
    cleanup : bool
        set the Publication date to type datetime
    show_info : bool
        some output
    empty_dataframe: bool
        removes empty lines
           
    Returns
    -------
    dataframe

    '''
    try:
        df=pd.read_csv(filename)
        
        if cleanup:
            df['Publication Date'] = pd.to_datetime(df['Publication Date'])
            if show_info:
                print(df.info())
            
            if empty_dataframe:
                df.dropna(inplace=True)
                if show_info:
                    print(df.info())

        return df
    except FileNotFoundError as not_found:
        print(f'File not found: {not_found.filename}')