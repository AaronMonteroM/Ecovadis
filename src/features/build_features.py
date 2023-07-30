import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn import preprocessing

def dataPreprocessing(df):
    """Processes the raw data before it is used in the predictive churn model

    Args:
        df (DataFrame): Raw data

    Returns:
        df: Transformed data
    """
    
    #Remove RowNumber, CustomerId and Surname
    df=df.iloc[:,3:]

    #Determine sentiment from customer feedback
    nltk.downloader.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    sent_comment=df['CustomerFeedback'].loc[df['CustomerFeedback'].isna()==False].apply(lambda x: sia.polarity_scores(x)["compound"])
    sent_comment_lvl=np.select([sent_comment < 0, sent_comment == 0, sent_comment > 0],
                               [-1,0,1])
    sent_comment_col=np.zeros(len(df))
    sent_comment_col[sent_comment.index]=sent_comment_lvl
    df['CustomerFeedback']=sent_comment_col

    #Categorical features to numerical ones
    le_country = preprocessing.LabelEncoder()
    df['Country']=le_country.fit_transform(df['Country'])
    le_gender = preprocessing.LabelEncoder()
    df['Gender']=le_gender.fit_transform(df['Gender'])
    
    return df
