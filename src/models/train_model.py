import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import sys
sys.path.append('../features')
from build_features import dataPreprocessing
#sys.path.append('../features')
#from features import build_features as f

def train_model(df):
    """Train the churn predictive model

    Args:
        df (DataFrame): Processed data

    Returns:
        pipe: trained pipe line that contains the churn predictive model
    """
    X=np.array(df.iloc[:,:-1]) #Features
    y=np.array(df.iloc[:,-1]) #Target class

    #Training model
    model=RandomForestClassifier(max_depth=16, min_samples_leaf=2, n_estimators=600, random_state=42)

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('sampling', RandomOverSampler(random_state=42)),
                     ('class', model)])

    pipe.fit(X,y)
    
    return pipe

if __name__ == '__main__':

    Nargs=len(sys.argv)
    if Nargs>2:
        raise Exception('Too many arguments')
    elif Nargs==1:
        raise Exception('Missing data file')
    else:
        df=pd.read_excel(sys.argv[1])
        try:
            df=dataPreprocessing(df)
        except:
            raise Exception("Missing features")
        model=train_model(df)
        dump(model, '../../models/churn_model.joblib')
