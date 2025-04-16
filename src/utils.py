"Have all the common imports in one place"
import os
import sys

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import dill

def save_object(file_path, obj):
    """
    Save the object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise Exception(f"Error saving object: {str(e)}")
    
def evaluate_model(x_train, y_train, x_test, y_test, models,params):
    """
    Evaluate the model and return the best model based on R2 score.
    """
    try:
        report = {}
        model_values = list(models.values())
        model_names = list(models.keys())
        for i in range(len(models)):
            model = model_values[i]
            para=params[model_names[i]]
            if model_names[i]=='SVR':
                rs=RandomizedSearchCV(model,para,cv=3,n_jobs=-1,n_iter=30)
            else:
                rs=RandomizedSearchCV(model,para,cv=3,n_jobs=-1)

            rs.fit(x_train,y_train)
            logging.info(f"{model} best params are : {rs.best_params_} and best score is: {rs.best_score_}")
            model.set_params(**rs.best_params_)
            model.fit(x_train,y_train)
            
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_names[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys) from e
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

    
