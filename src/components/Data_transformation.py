import os
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file: str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_features=['reading_score','writing_score']
            cat_features=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            cat_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(sparse_output=False)),
                ('scaler',StandardScaler())
            ])

            logging.info('Categroical columns: encoding completed')
            logging.info('Numerical columns scaling completed')

            preprocessor=ColumnTransformer([
                ("num", num_pipeline, num_features),
                ("cat", cat_pipeline, cat_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Train and Test data read completed')

            logging.info('Obtaining preprocessing object')

            preprocessor_obj=self.get_data_transformer_object()

            target_column_name='math_score'
            drop_columns=[target_column_name]
            num_features=['reading_score','writing_score']

            input_feature_train_df=train_df.drop(columns=drop_columns)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns)
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying preprocessing object on training and testing datasets')

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object')

            save_object(
                file_path=self.transformation_config.preprocessor_file,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_file
            )

        except Exception as e:
            raise CustomException(e, sys)