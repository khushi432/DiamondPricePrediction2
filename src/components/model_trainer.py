# Basic Import
import numpy as np
import pandas as pd
import sys
import os


# ML Models
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# Project Imports
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'DecisionTree':DecisionTreeRegressor(),
            # 'RandomForest': RandomForestRegressor(),
            # 'GradientBoosting':GradientBoostingRegressor()

            #  -------- Tuned Random Forest --------
                "RandomForest": RandomForestRegressor(
                    n_estimators=600,
                    max_depth=25,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=42
                ),

                # -------- Gradient Boosting --------
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                ),

                #  -------- XGBoost --------
                "XGBoost": XGBRegressor(
                    n_estimators=800,
                    learning_rate=0.05,
                    max_depth=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=42
                ),

                # -------- CatBoost (BEST) --------
                "CatBoost": CatBoostRegressor(
                    iterations=1500,
                    depth=8,
                    learning_rate=0.03,
                    loss_function="RMSE",
                    random_seed=42,
                    verbose=False
                )

        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)