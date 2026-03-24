import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer  # To handle missing values
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):      # This function is responsible for data transformation
        try:
            numerical_cols = ['reading score', 'writing score']
            categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='medium')),      # handles the missing values
                    ('scaler', StandardScaler())
                ]
            )
            logging.info("numerical columns scaline completed")      


            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy= 'most_frequent')),      # handles the missing values
                    ('One_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info("categorical columns encoding and scaling completed")


            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('cat_pipeline', cat_pipeline, categorical_cols)
                ]
            )
            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv("train_path")
            test_df = pd.read_csv('test_path')
            logging.info("Read Train & Test Data Completed")

            logging.info("Obtaining preprocessing object")

            preprocessor = self.get_data_transformer_object()

            target_column_name = 'math score'
            numerical_cols = ['reading score', 'writing score']
            categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            input_features_train_df = train_df.drop(target_column_name, axis = 1)
            input_featuers_test_df = test_df.drop(target_column_name, axis = 1)

            logging.info("Applying the preprocessing object on train and test data")
            

        except:
            pass