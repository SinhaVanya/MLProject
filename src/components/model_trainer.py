import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Linear Regression": LinearRegression(),
                "Lasso Regression": Lasso(),
                "Ridge Regression": Ridge(),
                "SVR": SVR(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Ada Boost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "LightGBM": LGBMRegressor()
            }

            #Evaluate models will be present in utils
            logging.info("Evaluating of models started")
            model_report:dict=evaluate_model(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,
                                             models=models)
            
            # Get best model score from the dictionary
            best_model_score=max(sorted(model_report.values()))

            # Get best model name from the dictionary
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2score=r2_score(predicted,y_test)
            return r2score
            
        except Exception as e:
            raise CustomException(e,sys)
