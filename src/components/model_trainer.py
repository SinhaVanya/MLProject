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
                "SVR": SVR(verbose=2),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Ada Boost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "LightGBM": LGBMRegressor()
            }

            params={
                "Linear Regression":{},
                "Lasso Regression": {
                    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 1, 10, 50, 100],
                    'fit_intercept': [True, False],
                    'selection': ['cyclic', 'random'],
                    'max_iter': [1000, 2000, 5000, 10000],
                    'tol': [1e-4, 1e-3, 1e-2]
                },
                "Ridge Regression": {
                    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 1, 10, 50, 100],
                    'fit_intercept': [True, False],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                    'max_iter': [1000, 2000, 5000, 10000],
                    'tol': [1e-4, 1e-3, 1e-2]
                    
                },
                "SVR": {
                    
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.01, 0.1, 1, 10],
                    'epsilon': [0.001, 0.01, 0.05, 0.1],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    #'degree': [2, 3, 4, 5],  # for 'poly' kernel
                    #'shrinking': [True, False]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                    'max_depth': [None, 5, 10, 20, 30, 40, 50, 75, 100],  # limits tree depth
                    'min_samples_split': [2, 5, 10, 15, 20, 50],  # min samples to split an internal node
                    'min_samples_leaf': [1, 2, 4, 6, 10, 20],  # miEn samples at a leaf node
                    'min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1],  # min weighted fraction of total sum of weights at leaf
                    'max_features': [None, 'sqrt', 'log2', 0.1, 0.3, 0.5, 0.7],  # number of features to consider for best split
                    'max_leaf_nodes': [None, 10, 20, 30, 40, 50, 100],  # limits leaf nodes
                    'min_impurity_decrease': [0.0, 0.01, 0.02, 0.05],  # early stopping criterion
                    'ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.05],  # post-pruning parameter (cost-complexity pruning)
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['auto', 'sqrt', 'log2', 0.1, 0.2, 0.3, 0.5, None],
                    'n_estimators': [8,16,32,64,128,256],
                    'max_depth': [None, 5, 10, 20, 30, 40, 50, 75, 100],  # control overfitting
                    'min_samples_split': [2, 5, 10, 15, 20, 50],  # minimum to split an internal node
                    'min_samples_leaf': [1, 2, 4, 6, 10, 20],  # minimum samples at a leaf
                    'max_leaf_nodes': [None, 10, 20, 30, 40, 50],  # limits leaf nodes
                    'bootstrap': [True, False],  # whether bootstrap sampling is used
                    'oob_score': [True, False],  # only when bootstrap=True
                    'ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.05],  # complexity pruning
                    'max_samples': [None, 0.3, 0.5, 0.7, 0.9],  # for subsampling in each tree (only if bootstrap=True)
                    'min_weight_fraction_leaf': [0.0, 0.01, 0.05],  # rarely used but helps in imbalance cases
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001, 0.2, 0.3],
                    'subsample':[0.5,0.6,0.7,0.75,0.8,0.85,0.9, 1.0],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256,100, 200, 300, 500, 800, 1000],
                    'max_depth': [3, 5, 7, 10, 15, 20],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 8, 10],
                    'max_features': ['auto', 'sqrt', 'log2', 0.2, 0.5, 0.7, 1.0],
                },
                "Ada Boost":{
                    'learning_rate':[.1,.01,0.5,.001, 0.05, 1.0],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256,50, 100, 200, 300, 500, 1000]
                                
                },
                "KNN": {
                    'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 20, 25, 30],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [10, 20, 30, 40, 50, 60],
                    'p': [1, 2, 3]  # 1 = Manhattan, 2 = Euclidean, 3 = Minkowski
                },
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001, 0.2],
                    'n_estimators': [8,16,32,64,128,256,100, 200, 300, 500, 800],
                    'max_depth': [3, 5, 7, 10, 15],
                    'min_child_weight': [1, 3, 5, 7],
                    'subsample': [0.5, 0.7, 0.8, 1.0],
                    'colsample_bytree': [0.5, 0.7, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.3, 0.5, 1.0],
                    'reg_alpha': [0, 0.01, 0.1, 1, 10],
                    'reg_lambda': [0, 0.01, 0.1, 1, 10],
                    'scale_pos_weight': [1, 2, 5],  # useful in imbalanced regression targets
                    'booster': ['gbtree', 'gblinear', 'dart']
                },
                "LightGBM": {
                    'n_estimators': [100, 200, 500, 800, 1000],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                    'max_depth': [-1, 3, 5, 7, 10, 15, 20],
                    'num_leaves': [20, 31, 40, 60, 80, 100],
                    'min_child_samples': [5, 10, 20, 30, 50],
                    'subsample': [0.5, 0.6, 0.8, 1.0],
                    'colsample_bytree': [0.5, 0.6, 0.8, 1.0],
                    'reg_alpha': [0.0, 0.01, 0.1, 0.5, 1.0],
                    'reg_lambda': [0.0, 0.01, 0.1, 0.5, 1.0],
                    'boosting_type': ['gbdt', 'dart', 'goss']
                }   
            }

            #Evaluate models will be present in utils
            logging.info("Evaluating of models started")
            model_report:dict=evaluate_model(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,
                                             models=models,params=params)
            
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
