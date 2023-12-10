import os
import sys
import time
import glob
import pdb
sys.path.append(r"/home/tarunpreet/")
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from time import process_time
import seaborn as sns
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import multiprocessing as mp
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.svm import SVR
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score,r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.svm import SVC
from matplotlib import pyplot
from Features_ext import *
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression,HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from numpy import arange
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import VotingRegressor
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.tree import DecisionTreeRegressor
import pickle
from joblib import Parallel, delayed
from sklearn.ensemble import StackingRegressor
import joblib
  


if __name__=='__main__':

	
	t1_start=process_time()
	path="data0"
	all_files=glob.glob(path+"/*.csv") 
	all_files=glob.glob(os.path.join(path,"*.csv"))
	F_data2=pd.DataFrame()
	
	for filename in all_files:
		data=pd.read_csv(filename)
		outcome=str(data['Outcome'][0])
		FN=filename.split("/")[1]
		X_data1=data.drop(labels=['Date','Time','ICU_Stay','Hospital_Stay','Outcome','IBPS','IBPD'],axis=1)
		X_data1=X_data1.replace('---',np.nan)
		X_data1=X_data1.replace('N/R',np.nan)
		X_data2=X_data1.replace(np.nan,0)
		data.loc[(data['Hospital_Stay'] < 0) , 'Hospital_Stay'] = np.nan
		data=data.replace(np.nan,0)
		
		#Define y dataset
		y=(data['Hospital_Stay'][0])
		F_data=features_extraction(X_data2)
		F_data['Hospital_Stay']=y
		F_data['case_number']=FN
		F_data1=F_data.replace([np.inf, -np.inf,np.nan,'j','None'], 0)
		F_data2=pd.concat([F_data1,F_data2],axis=0)
	
	F_data2=F_data2.replace([np.inf, -np.inf,np.nan,'j','None'], 0)
	F_data2['Hospital_Stay']=F_data2['Hospital_Stay'].astype(float)
	case=F_data2['case_number'].to_numpy()

	#Define X dataset
	X_train=F_data2.drop(labels=['Hospital_Stay','case_number'],axis=1)
	print(X_train)
	#Define Y dataset
	Y_train=F_data2['Hospital_Stay']
	y_train=Y_train.to_numpy().astype(float)
	
	#scale the feature extraction dataset before hyperparameter optimization
	scaler=RobustScaler()
	scaler.fit(X_train)
	SX_train=scaler.transform(X_train)
	XA_train=pd.DataFrame(SX_train)
	joblib.dump(scaler, 'scaler.pkl')
	
	# test
	path="test_data"
	all_files=glob.glob(path+"/*.csv") 
	all_files=glob.glob(os.path.join(path,"*.csv"))
	F_data2=pd.DataFrame()
	for filename in all_files:
		data=pd.read_csv(filename)
		column_name_mapping = {"HR_bpm_":"HR(BPM)","IBP_M_":"IBPM","IBP_S_":'IBPS',"IBP_D_":'IBPD'}
		data = data.rename(columns=column_name_mapping)
		outcome=str(data['Outcome'][0])
		FN=filename.split("/")[1]
		print(FN)
		# Define X dataset
		X_data1=data.drop(labels=['Date','Time','ICU_Stay','Hospital_Stay','Outcome','IBPS','IBPD'],axis=1)
		X_data1=X_data1.replace('---',np.nan)
		X_data1=X_data1.replace('N/R',np.nan)
		X_data2=X_data1.replace(np.nan,0)
		#data['Hospital_Stay'] = pd.to_numeric(data['Hospital_Stay'], errors='coerce')

		#data.loc[(data['Hospital_Stay'] < 0) , 'Hospital_Stay'] = np.nan
		data=data.replace(np.nan,0)
		
		#Define y dataset
		y=(data['Hospital_Stay'][0])
		print(y)
		F_data=features_extraction(X_data2)
		F_data['Hospital_Stay']=y
		F_data['case_number']=FN
		F_data1=F_data.replace([np.inf, -np.inf,np.nan,'j','None'], 0)
		F_data2=pd.concat([F_data1,F_data2],axis=0)
	
	F_data2=F_data2.replace([np.inf, -np.inf,np.nan,'j','None'], 0)
	F_data2['Hospital_Stay']=F_data2['Hospital_Stay'].str.replace('[\[\]\(\),:\'’’""]', '')
	F_data2['Hospital_Stay']=F_data2['Hospital_Stay'].astype(float)
	case=F_data2['case_number'].to_numpy()

	#Define X dataset
	X_test=F_data2.drop(labels=['Hospital_Stay','case_number'],axis=1)
	print(X_test)
	#Define Y dataset
	Y_test=F_data2['Hospital_Stay']
	y_test=Y_test.to_numpy().astype(float)
	scaler= joblib.load('scaler.pkl')
	SX1_test=scaler.transform(X_test)
	XA_test=pd.DataFrame(SX1_test)
		
	
	cv = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
	# Random forest Regessor
	param_rf = {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6,8],'bootstrap':[False],'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
	rf_reg = GridSearchCV(RandomForestRegressor(), param_rf, cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
		
	#XGBoost Regressor
	param_xgb = {'n_estimators': [100, 80, 60, 55, 51, 45], 'max_depth': [7, 8], 'reg_lambda' :[0.26, 0.25, 0.2]}
	XGB_reg=GridSearchCV(xgb.XGBRFRegressor(), param_xgb, scoring='neg_mean_absolute_error', cv=3, return_train_score=True,n_jobs=-1)
	
	param_gbr = {'learning_rate': [0.0001, 0.01,0.001],'n_estimators' : [10,20,30,40,50,60,70,80,90,100]}
	GBR_reg=GridSearchCV(GradientBoostingRegressor(), param_gbr, scoring='neg_mean_absolute_error', cv=3, return_train_score=True,n_jobs=-1)
	
	optimized_rf_reg=rf_reg.fit(XA_train,y_train)
	print("Best: %f using %s for %s" % (optimized_rf_reg.best_score_,optimized_rf_reg.best_params_,optimized_rf_reg.best_estimator_))
		
	optimized_xgb_reg=XGB_reg.fit(XA_train,y_train)
	print("Best: %f using %s for %s" % (optimized_xgb_reg.best_score_,optimized_xgb_reg.best_params_,optimized_xgb_reg.best_estimator_))
		
	optimized_gbr_reg=GBR_reg.fit(XA_train,y_train)
	print("Best: %f using %s for %s" % (optimized_gbr_reg.best_score_,optimized_gbr_reg.best_params_,optimized_gbr_reg.best_estimator_))
	best= [optimized_rf_reg.best_estimator_,optimized_xgb_reg.best_estimator_,optimized_gbr_reg.best_estimator_]
	m1=optimized_rf_reg.best_estimator_
	print("model used is",m1)
	m1.fit(XA_train, y_train)  
	joblib.dump(m1, 'WithoutdriftHS_RF.pkl')	
	loaded_model=joblib.load('WithoutdriftHS_RF.pkl')
	Pred = loaded_model.predict(XA_test)
	Predictions=Pred.astype(int)
	Predictions=abs(Predictions)
	Actual=y_test
	
	df=pd.DataFrame(Actual,Predictions)
	print("Normal Approach")
	print('Actual values',y_test)
	print("Predicted values",Predictions)
	print('MAE:', metrics.mean_absolute_error(Actual, Predictions))
	print('MSE:', metrics.mean_squared_error(Actual,Predictions))
	print('RMSE:', np.sqrt(metrics.mean_squared_error(Actual, Predictions)))
	print('R2_score:', metrics.r2_score(Actual, Predictions))
	
	m2=optimized_xgb_reg.best_estimator_
	print("model used is",m2)
	m2.fit(XA_train, y_train)  
	joblib.dump(m2, 'WithoutdriftHS_XG.pkl')	
	loaded_model=joblib.load('WithoutdriftHS_XG.pkl')
	Pred = loaded_model.predict(XA_test)
	Predictions=Pred.astype(int)
	Predictions=abs(Predictions)
	Actual=y_test
	
	df=pd.DataFrame(Actual,Predictions)
	print("Normal Approach")
	print('Actual values',y_test)
	print("Predicted values",Predictions)
	print('MAE:', metrics.mean_absolute_error(Actual, Predictions))
	print('MSE:', metrics.mean_squared_error(Actual,Predictions))
	print('RMSE:', np.sqrt(metrics.mean_squared_error(Actual, Predictions)))
	print('R2_score:', metrics.r2_score(Actual, Predictions))
	
	m3=optimized_gbr_reg.best_estimator_
	print("model used is",m3)
	m3.fit(XA_train, y_train)  
	joblib.dump(m3, 'WithoutdriftHS_GB.pkl')	
	loaded_model=joblib.load('WithoutdriftHS_GB.pkl')
	Pred = loaded_model.predict(XA_test)
	Predictions=Pred.astype(int)
	Predictions=abs(Predictions)
	Actual=y_test
	
	df=pd.DataFrame(Actual,Predictions)
	print("Normal Approach")
	print('Actual values',y_test)
	print("Predicted values",Predictions)
	print('MAE:', metrics.mean_absolute_error(Actual, Predictions))
	print('MSE:', metrics.mean_squared_error(Actual,Predictions))
	print('RMSE:', np.sqrt(metrics.mean_squared_error(Actual, Predictions)))
	print('R2_score:', metrics.r2_score(Actual, Predictions))
	
		
	# new data
	path="data5"
	all_files=glob.glob(path+"/*.csv") 
	all_files=glob.glob(os.path.join(path,"*.csv"))
	F_data2=pd.DataFrame()
	
	for filename in all_files:
		data=pd.read_csv(filename)
		outcome=str(data['Outcome'][0])
		FN=filename.split("/")[1]
		
		# Define X dataset
		column_name_mapping = {"HR_bpm_":"HR(BPM)","IBP_M_":"IBPM","IBP_S_":'IBPS',"IBP_D_":'IBPD'}
		
		data = data.rename(columns=column_name_mapping)
		columns=['Date','Time','ICU_Stay','Hospital_Stay','Outcome','IBPS','IBPD']
		X_data1=data.drop(columns=columns)
		X_data1=X_data1.replace('---',np.nan)
		X_data1=X_data1.replace('N/R',np.nan)
		X_data2=X_data1.replace(np.nan,0)
		
		data=data.replace(np.nan,0)
		print(filename)
		#Define y dataset
		y=(data['Hospital_Stay'][0])
		F_data=features_extraction(X_data2)
		F_data['Hospital_Stay']=y
		F_data['case_number']=FN
		F_data1=F_data.replace([np.inf, -np.inf,np.nan,'j','None'], 0)
		F_data2=pd.concat([F_data1,F_data2],axis=0)
	
	F_data2['Hospital_Stay']=F_data2['Hospital_Stay'].str.replace('[\[\]\(\),:\'’’""]', '')
	F_data2=F_data2.replace([np.inf, -np.inf,np.nan,'j','None'], 0)
	F_data2['Hospital_Stay']=F_data2['Hospital_Stay'].astype(float)
	case=F_data2['case_number'].to_numpy()

	
	#Define X dataset
	X_ntrain=F_data2.drop(labels=['Hospital_Stay','case_number'],axis=1)
	
	#Define Y dataset
	Y_ntrain=F_data2['Hospital_Stay']
	y_ntrain=Y_ntrain.to_numpy().astype(float)
	
	#scale the feature extraction dataset before hyperparameter optimization
	scaler=RobustScaler()
	scaler.fit(X_ntrain)
	SX_ntrain=scaler.transform(X_ntrain)
	XA_ntrain=pd.DataFrame(SX_ntrain)
	joblib.dump(scaler, 'scalernew.pkl')
	
	#normal approach 
	
	
	
	
	# adaptive approach
	X_ftrain=pd.concat([XA_train, XA_ntrain], ignore_index=True)
	y_ftrain=np.concatenate((y_train, y_ntrain), axis=0)
	cv = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
	param_gbr = {'learning_rate': [0.0001, 0.01,0.001],'n_estimators' : [10,20,30,40,50]}
	GBR_reg=GridSearchCV(GradientBoostingRegressor(), param_gbr, scoring='neg_mean_absolute_error', cv=3, return_train_score=True,n_jobs=-1)
	optimized_gbr_reg=GBR_reg.fit(X_ftrain,y_ftrain)
	print("Best: %f using %s for %s" % (optimized_gbr_reg.best_score_,optimized_gbr_reg.best_params_,optimized_gbr_reg.best_estimator_))
	best= [optimized_gbr_reg.best_estimator_]
	for m in best:
		print("model used is",m)
		m.fit(X_ftrain, y_ftrain)  
		joblib.dump(m, 'Adaptive_HS.pkl')	
	loaded_model=joblib.load('Adaptive_HS.pkl')
	Pred = loaded_model.predict(XA_test)
	Predictions=Pred.astype(int)
	Predictions=abs(Predictions)
	Actual=y_test
	
	df=pd.DataFrame(Actual,Predictions)
	print("Adaptive Approach")
	print('Actual values',y_test)
	print("Predicted values",Predictions)
	print('MAE:', metrics.mean_absolute_error(Actual, Predictions))
	print('MSE:', metrics.mean_squared_error(Actual,Predictions))
	print('RMSE:', np.sqrt(metrics.mean_squared_error(Actual, Predictions)))
	print('R2_score:', metrics.r2_score(Actual, Predictions))
	
	
	#Incremental Approach
	cv = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
	# Random forest Regessor
	param_rf = {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6,8],'bootstrap':[False],'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
	rf_reg = GridSearchCV(RandomForestRegressor(), param_rf, cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
		
	#XGBoost Regressor
	param_xgb = {'n_estimators': [100, 80, 60, 55, 51, 45], 'max_depth': [7, 8], 'reg_lambda' :[0.26, 0.25, 0.2]}
	XGB_reg=GridSearchCV(xgb.XGBRFRegressor(), param_xgb, scoring='neg_mean_absolute_error', cv=3, return_train_score=True,n_jobs=-1)
	
	param_gbr = {'learning_rate': [0.0001, 0.01,0.001],'n_estimators' : [10,20,30,40,50,60,70,80,90,100]}
	GBR_reg=GridSearchCV(GradientBoostingRegressor(), param_gbr, scoring='neg_mean_absolute_error', cv=3, return_train_score=True,n_jobs=-1)
	
	optimized_rf_reg=rf_reg.fit(XA_ntrain,y_ntrain)
	print("Best: %f using %s for %s" % (optimized_rf_reg.best_score_,optimized_rf_reg.best_params_,optimized_rf_reg.best_estimator_))
		
	optimized_xgb_reg=XGB_reg.fit(XA_ntrain,y_ntrain)
	print("Best: %f using %s for %s" % (optimized_xgb_reg.best_score_,optimized_xgb_reg.best_params_,optimized_xgb_reg.best_estimator_))
		
	optimized_gbr_reg=GBR_reg.fit(XA_ntrain,y_ntrain)
	print("Best: %f using %s for %s" % (optimized_gbr_reg.best_score_,optimized_gbr_reg.best_params_,optimized_gbr_reg.best_estimator_))
	best= [optimized_rf_reg.best_estimator_,optimized_xgb_reg.best_estimator_,optimized_gbr_reg.best_estimator_]
	m1=optimized_rf_reg.best_estimator_
	
	m1.fit(XA_ntrain, y_ntrain)  
	joblib.dump(m1, 'IncrementalHS_RF.pkl')	
	
	m2=optimized_xgb_reg.best_estimator_
	
	m2.fit(XA_ntrain, y_ntrain)  
	joblib.dump(m2, 'IncrementalHS_XG.pkl')	
	
	m3=optimized_gbr_reg.best_estimator_
	
	m3.fit(XA_ntrain, y_ntrain)  
	joblib.dump(m3, 'IncrementalHS_GB.pkl')	
	
	
	loaded_model=joblib.load('IncrementalHS_GB.pkl')
	loaded_model.set_params(n_estimators=100,warm_start=True)
	loaded_model.fit(XA_ntrain,y_ntrain)
	Pred = loaded_model.predict(XA_test)
	Predictions=Pred.astype(int)
	Predictions=abs(Predictions)
	Actual=y_test
	df=pd.DataFrame(Actual,Predictions)
	print("Incremental Approach")
	print('Actual values',y_test)
	print("Predicted values",Predictions)
	print('MAE:', metrics.mean_absolute_error(Actual, Predictions))
	print('MSE:', metrics.mean_squared_error(Actual,Predictions))
	print('RMSE:', np.sqrt(metrics.mean_squared_error(Actual, Predictions)))
	print('R2_score:', metrics.r2_score(Actual, Predictions))
	
	
    	# ensemble approach
	model1 = joblib.load('WithoutdriftHS_RF.pkl')
	model2= joblib.load('IncrementalHS_RF.pkl')
	eclf1 = VotingRegressor(estimators=[('model1',model1), ('model2',model2)])
	eclf1.fit(XA_ntrain, y_ntrain)
	
	Actual=y_test
	
	pred1= eclf1.predict(XA_test)	
	Predictions1=pred1.astype(int)
	Predictions1=abs(Predictions1)
	print("voting ensemble RF")
	print("model used is",m1)
	print('Actual values',y_test)
	print("Predicted values",Predictions1)
	print('MAE:', metrics.mean_absolute_error(Actual, Predictions1))
	print('MSE:', metrics.mean_squared_error(Actual,Predictions1))
	print('RMSE:', np.sqrt(metrics.mean_squared_error(Actual, Predictions1)))
	print('R2_score:', metrics.r2_score(Actual, Predictions1))
	
	model1 = joblib.load('WithoutdriftHS_XG.pkl')
	model2= joblib.load('IncrementalHS_XG.pkl')
	eclf1 = VotingRegressor(estimators=[('model1',model1), ('model2',model2)])
	eclf1.fit(XA_ntrain, y_ntrain)
	pred1= eclf1.predict(XA_test)	
	Predictions1=pred1.astype(int)
	Predictions1=abs(Predictions1)
	print("voting ensemble XG")
	print("model used is",m2)
	print('Actual values',y_test)
	print("Predicted values",Predictions1)
	print('MAE:', metrics.mean_absolute_error(Actual, Predictions1))
	print('MSE:', metrics.mean_squared_error(Actual,Predictions1))
	print('RMSE:', np.sqrt(metrics.mean_squared_error(Actual, Predictions1)))
	print('R2_score:', metrics.r2_score(Actual, Predictions1))
	
	
	model1 = joblib.load('WithoutdriftHS_GB.pkl')
	model2= joblib.load('IncrementalHS_GB.pkl')
	eclf1 = VotingRegressor(estimators=[('model1',model1), ('model2',model2)])
	eclf1.fit(XA_ntrain, y_ntrain)
	
	Actual=y_test
	
	pred1= eclf1.predict(XA_test)	
	Predictions1=pred1.astype(int)
	Predictions1=abs(Predictions1)
	print("voting ensemble GB")
	print("model used is",m3)
	print('Actual values',y_test)
	print("Predicted values",Predictions1)
	print('MAE:', metrics.mean_absolute_error(Actual, Predictions1))
	print('MSE:', metrics.mean_squared_error(Actual,Predictions1))
	print('RMSE:', np.sqrt(metrics.mean_squared_error(Actual, Predictions1)))
	print('R2_score:', metrics.r2_score(Actual, Predictions1))
	
