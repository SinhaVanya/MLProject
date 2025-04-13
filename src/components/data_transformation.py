import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Any input that is required for the data transformation process can be defined here.
    This includes the file paths for the preprocessor object and the transformed data.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This function creates a preprocessor object that handles the transformation of numerical and categorical data.
        It uses pipelines to handle missing values and scaling for numerical columns, and encoding for categorical columns.
        """

        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            #Created Pipeline handling missing values and scaling for numerical columns
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            #Created Pipeline handling missing values and encoding for categorical columns
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore', drop='first')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]

            )
            return preprocessor 
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read Train and Test data completed")

            logging.info("Obtaining Preprocessing Object")
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name='math_score'
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing dataframes")
            input_features_train_df_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_df_arr=preprocessing_obj.transform(input_features_test_df)

            #Concatenationg the transformed input features and target feature for train and test dataframes
            train_arr = np.c_[input_features_train_df_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_features_test_df_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved Preprocessing Object")
            #sav_object is in utils.py that is saving the preprocessor pickle file in the artifacts folder
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)