import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def average_score(df,years, newspaper, score_column:str):
    """
    prints the average of the dataframe for the given column

    Parameters
    ----------
    df: pd.DataFrame
        the dataframe to work with
    years:list|str
        either a list of years [2012,2013]  or 'ALL'
    newspaper:str
        either a valid newspaper or 'ALL'
    score_column : sttr
        The column to use for mean value 

    """
    # Filter the DataFrame based on 'newspaper' and 'years', if they are not 'ALL'
    if newspaper != 'ALL':
        df_filtered = df[df['Newspaper'] == newspaper]
    else:
        df_filtered = df

    if years != 'ALL':
        df_filtered = df_filtered[df_filtered['Year'].isin(years)]
    
    # Calculate the average score
    average_score_value = df_filtered[score_column].mean()
    
    # Print out the results in a formatted string
    newspaper_text = f"for {newspaper}" if newspaper != 'ALL' else "for all newspapers"
    years_text = f"in the year(s) {', '.join(map(str, years))}" if years != 'ALL' else "for all years"
    print(f"The average {score_column} score {newspaper_text} {years_text} is: {average_score_value:.2f}")

def show_plt_year(df,start_year, end_year, column='Sentiment_Score'):
    # Filter data between start_year and end_year and remove articles with a sentiment score of zero
    df_filtered = df[
        (df['Year'] >= start_year) & 
        (df['Year'] <= end_year) & 
        (df[column] != 0)
    ]
    
    # Ensure the 'Publication Date' is in datetime format
    #df_filtered['Publication Date'] = pd.to_datetime(df_filtered['Publication Date'])

    # Scatter plot with each article as a point
    plt.figure(figsize=(14, 7))
    plt.scatter(df_filtered['Publication Date'], df_filtered[column], color='blue', alpha=0.5)

    # Fit a polynomial trend line of degree 1 (linear trend line)
    z = np.polyfit(mdates.date2num(df_filtered['Publication Date']), df_filtered[column], 1)  # linear regression
    p = np.poly1d(z)

    # Plotting the trend line
    plt.plot(df_filtered['Publication Date'], p(mdates.date2num(df_filtered['Publication Date'])), "r--")

    # Formatting the plot
    plt.title(f'Sentiment Score for {start_year}-{end_year}')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.grid(True)
    plt.tight_layout()

    plt.show()