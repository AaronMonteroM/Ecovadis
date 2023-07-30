import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from joblib import load
import sys
sys.path.append('../features')
from build_features import dataPreprocessing

def predict_model(df):
    """Churn prediction on input data

    Args:
        df (DataFrame): Processed data

    Returns:
        y_pred: array with the churn prediction
    """
    
    model=load('../../models/churn_model.joblib')
    
    if len(df.columns)==12: #There is Exited column
        X=np.array(df.iloc[:,:-1]) #Features
        y_true=np.array(df.iloc[:,-1]) #Target class
        y_pred=model.predict(X)
        test_score=round(f1_score(y_true, y_pred, average='weighted'),2)
        print('F1 score for test:'+str(test_score))
        
    elif len(df.columns)==11: #Only features
        X=np.array(df)
        y_pred=model.predict(X)
        
    else: #Wrong dimensions
        raise Exception("Wrong number of features")
    
    return y_pred

if __name__ == '__main__':
    
    Nargs=len(sys.argv)
    if Nargs>2:
        raise Exception('Too many arguments')
    elif Nargs==1:
        raise Exception('Missing data file')
    else:
        df=pd.read_excel(sys.argv[1])
        try:
            df_transformed=dataPreprocessing(df)
            y_pred=predict_model(df_transformed)
            try:
                df_pred=pd.DataFrame(data={'CustomerId':df['CustomerId'].astype(int),
                                           'Exited_pred':y_pred.astype(int)})
                df_pred.to_csv('../../data/processed/churn_predicted.csv',index=False)
            except:
                raise Exception("Missing CustomerId")
        except:
            raise Exception("Missing features")
