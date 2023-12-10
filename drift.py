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
import statsmodels.api
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
#from imblearn.over_sampling import SMOTE
from sklearn.svm import SVR
#import xgboost as xgb
from sklearn import metrics
#from sklearn.metrics import roc_

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
#from Features_ext import *
#from statsmodels.tsa.stattools import adfuller
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
from sklearn.neighbors import LocalOutlierFactor as LOF
import pickle
import numpy as np
from scipy.stats import entropy
from scipy import stats
from scipy.spatial import distance
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from skmultiflow.drift_detection import ADWIN
from river.drift import PageHinkley
from scipy.stats import fisher_exact
from scipy.stats import mannwhitneyu


def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 10

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)
    
    
def data_length_normalizer(gt_data, obs_data, bins = 100):
    """
    Data length normalizer will normalize a set of data points if they
    are not the same length.
    
    params:
        gt_data (List) : The list of values associated with the training data
        obs_data (List) : The list of values associated with the observations
        bins (Int) : The number of bins you want to use for the distributions
        
    returns:
        The ground truth and observation data in the same length.
    """

    if len(gt_data) == len(obs_data):
        return gt_data, obs_data 

    # scale bins accordingly to data size
    if (len(gt_data) > 20*bins) and (len(obs_data) > 20*bins):
        bins = 10*bins 

    # convert into frequency based distributions
    gt_hist = plt.hist(gt_data, bins = bins)[0]
    obs_hist = plt.hist(obs_data, bins = bins)[0]
    plt.close()  # prevents plot from showing
    return gt_hist, obs_hist 

def softmax(vec):
    """
    This function will calculate the softmax of an array, essentially it will
    convert an array of values into an array of probabilities.
    
    params:
        vec (List) : A list of values you want to calculate the softmax for
        
    returns:
        A list of probabilities associated with the input vector
    """
    return(np.exp(vec)/np.exp(vec).sum())

def calc_cross_entropy(p, q):
    """
    This function will calculate the cross entropy for a pair of 
    distributions.
    
    params:
        p (List) : A discrete distribution of values
        q (List) : Sequence against which the relative entropy is computed.
        
    returns:
        The calculated entropy
    """
    return entropy(p,q)
    
def calc_drift(gt_data, obs_data, gt_col, obs_col):
    """
    This function will calculate the drift of two distributions given
    the drift type identifeid by the user.
    
    params:
        gt_data (DataFrame) : The dataset which holds the training information
        obs_data (DataFrame) : The dataset which holds the observed information
        gt_col (String) : The training data column you want to compare
        obs_col (String) : The observation column you want to compare
        
    returns:
        A drift score
    """

    gt_data = gt_data[gt_col].values
    obs_data = obs_data[obs_col].values

    # makes sure the data is same size
    gt_data, obs_data = data_length_normalizer(
        gt_data = gt_data,
        obs_data = obs_data
    )

    # convert to probabilities
    gt_data = softmax(gt_data)
    obs_data = softmax(obs_data)

    # run drift scores
    drift_score = calc_cross_entropy(gt_data, obs_data)
    return drift_score
   
def cdf(sample, x, sort = False):
    # Sorts the sample, if unsorted
    if sort:
        sample.sort()
    # Counts how many observations are below x
    cdf = sum(sample <= x)
    # Divides by the total number of observations
    cdf = cdf / len(sample)
    return cdf
    
def ks_2samp(sample1, sample2):
    # Gets all observations
    observations = np.concatenate((sample1, sample2))
    observations.sort()
    # Sorts the samples
    sample1.sort()
    sample2.sort()
    # Evaluates the KS statistic
    D_ks = [] # KS Statistic list
    for x in observations:
        cdf_sample1 = cdf(sample = sample1, x  = x)
        cdf_sample2 = cdf(sample = sample2, x  = x)
        D_ks.append(abs(cdf_sample1 - cdf_sample2))
    ks_stat = max(D_ks)
    # Calculates the P-Value based on the two-sided test
    # The P-Value comes from the KS Distribution Survival Function (SF = 1-CDF)
    m, n = float(len(sample1)), float(len(sample2))
    en = m * n / (m + n)
    p_value = stats.kstwo.sf(ks_stat, np.round(en))
    return {"ks_stat": ks_stat, "p_value" : p_value}
def jsd(p, q, base=2):
    '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return scipy.stats.entropy(p,m, base=base)/2. +  scipy.stats.entropy(q, m, base=base)/2
def js(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
   # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    divergence= (entropy(p, m) + entropy(q, m)) / 2
    distance = np.sqrt(divergence)
    return distance

    
def KL(P,Q):

     epsilon = 0.00001

     # You may want to instead make copies to avoid changing the np arrays.
     P = P+epsilon
     Q = Q+epsilon

     divergence = np.sum(P*np.log(P/Q))
     return divergence
     
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

if __name__=='__main__':

	t1_start=process_time()

	path="data0"
	all_files=glob.glob(path+"/*.csv") 
	all_files=glob.glob(os.path.join(path,"*.csv"))
	F_data2=pd.DataFrame()
	F_data3=pd.DataFrame()
	
	for filename in all_files:
		data=pd.read_csv(filename)
		outcome=str(data['Outcome'][0])
		FN=filename.split("/")[1]
		F_data=data.drop(labels=['Date','Time','IBPS','IBPD','Outcome'],axis=1)
		F_data[F_data<= 0] = np.nan
		F_data0=F_data.replace(np.nan,0)
		F_data0['HR']=F_data0['HR(BPM)'].mean()
		F_data0['IBP']=F_data0['IBPM'].mean()
		

		F_data0['ICU_Stay']=(F_data['ICU_Stay'][0])
		F_data0['Hospital_Stay']=(F_data['Hospital_Stay'][0])
		F_data0['case_number']=FN
		F_data1=pd.DataFrame(F_data0,index=[0])
		
		
		F_data2=pd.concat([F_data1,F_data2],axis=0)
	F_data2.loc[(F_data2['ICU_Stay'] <= 0)] = np.nan
	F_data2.loc[(F_data2['Hospital_Stay']<=0)]=np.nan
	F_data2=F_data2.dropna(axis=0)
	
	
	Y_train_ICU=F_data2['ICU_Stay']
	Y_train_HS=F_data2['Hospital_Stay']
	y_train_ICU=Y_train_ICU.to_numpy().astype(float)
	y_train_HS=Y_train_HS.to_numpy().astype(float)
	maximum_ICU=np.max(y_train_ICU)
	minimum_ICU=np.min(y_train_ICU)
	maximum_HS=np.max(y_train_HS)
	minimum_HS=np.min(y_train_HS)
	
	y_train_HR=F_data2['HR']
	y_train_IBP=F_data2['IBP']
	maximum_HR=np.max(y_train_HR)
	minimum_HR=np.min(y_train_HR)
	maximum_IBP=np.max(y_train_IBP)
	minimum_IBP=np.min(y_train_IBP)
	
	
	path="data5"
	all_files=glob.glob(path+"/*.csv") 
	all_files=glob.glob(os.path.join(path,"*.csv"))
	
	
	for filename in all_files:
		data=pd.read_csv(filename)
		outcome=str(data['Outcome'][0])
		FN=filename.split("/")[1]
		F_data=data.drop(labels=['Date','Time','IBPS','IBPD','Outcome'],axis=1)
		F_data[F_data<= 0] = np.nan
		F_data0=F_data.replace(np.nan,0)
		F_data0['HR']=F_data0['HR(BPM)'].mean()
		F_data0['IBP']=F_data0['IBPM'].mean()
		F_data0['ICU_Stay']=(F_data['ICU_Stay'][0])
		F_data0['Hospital_Stay']=(F_data['Hospital_Stay'][0])
		F_data0['case_number']=FN
		F_data1=pd.DataFrame(F_data0,index=[0])
		F_data1['ICU_Stay']=F_data1['ICU_Stay'].astype(float)
		F_data1['Hospital_Stay']=F_data1['Hospital_Stay'].astype(float)
		
		#F_data1['HR']=F_data1['
		
		
		
		if ((F_data1['ICU_Stay']<minimum_ICU )| (F_data1['ICU_Stay']>maximum_ICU)).all():
			print("case no." ,FN,"has drift with ICU stay value of", F_data1['ICU_Stay'])
		else :
			print("case no." ,FN,"has no drift in ICU Stay")
		
		if ((F_data1['Hospital_Stay']<minimum_HS )| (F_data1['Hospital_Stay']>maximum_HS)).all():
			print("case no." ,FN,"has drift with Hospital stay value of", F_data1['Hospital_Stay'])
		else :
			print("case no." ,FN,"has no drift in Hospital Stay")
		if ((F_data1['HR']<minimum_HR )| (F_data1['HR']>maximum_HR)).all():
			print("case no." ,FN,"has drift with Heart Rate value of", F_data1['HR'])
		else :
			print("case no." ,FN,"has no drift in Heart Rate")
		if ((F_data1['IBP']<minimum_IBP )| (F_data1['IBP']>maximum_IBP)).all():
			print("case no." ,FN,"has drift with Blood Pressure value of", F_data1['IBP'])
		else :
			print("case no." ,FN,"has no drift in Blood Pressure")
		print()
		
		F_data3=pd.concat([F_data3,F_data1],axis=0)
	F_data3.loc[(F_data3['ICU_Stay'] <= 0)] = np.nan
	F_data3.loc[(F_data3['Hospital_Stay'] <= 0)] = np.nan
	
	F_data3=F_data3.dropna(axis=0)
	Y_test_ICU=F_data3['ICU_Stay']
	y_test_ICU=Y_test_ICU.to_numpy().astype(float)
	y_trainICU=y_train_ICU/y_train_ICU.sum(axis=0)
	y_testICU=y_test_ICU/y_test_ICU.sum(axis=0)
	Y_test_HS=F_data3['Hospital_Stay']
	y_test_HS=Y_test_HS.to_numpy().astype(float)
	y_trainHS=y_train_HS/y_train_HS.sum(axis=0)
	y_testHS=y_test_HS/y_test_HS.sum(axis=0)
	y_test_HR=F_data3['HR']
	y_test_IBP=F_data3['IBP']
	maximum_ICU=np.max(y_test_ICU)
	minimum_ICU=np.min(y_test_ICU)
	maximum_HS=np.max(y_test_HS)
	minimum_HS=np.min(y_test_HS)
	print("Statistical Tests")
	
	
	print("Population stability index for ICU Stay is",calculate_psi(y_train_ICU,y_test_ICU))
	print("Population stability index for Hospital Stay is",calculate_psi(y_train_HS,y_test_HS))
	print("Population stability index for Heart Rate is",calculate_psi(y_train_HR,y_test_HR))
	print("Population stability index for Blood Pressure is",calculate_psi(y_train_IBP,y_test_IBP))
	print()	
	#print("Drift score in ICU Stay", calc_drift(gt_data = F_data2, obs_data = F_data3, gt_col = 'ICU_Stay', obs_col = 'ICU_Stay'))
	#print("Drift score in Hospital Stay", calc_drift(gt_data = F_data2, obs_data = F_data3, gt_col = 'Hospital_Stay', obs_col = 'Hospital_Stay'))
	p_value = 0.05
	print()
	rejected = 0
	test_ICU={}
	test_HS={}
	test_ICU = stats.ks_2samp(y_train_ICU, y_test_ICU)
	test_HS = stats.ks_2samp(y_train_HS, y_test_HS)
	test_HR = stats.ks_2samp(y_train_HR, y_test_HR)
	test_IBP = stats.ks_2samp(y_train_IBP, y_test_IBP)
	res_ICU=stats.anderson_ksamp([y_train_ICU, y_test_ICU])
	res_HS=stats.anderson_ksamp([y_train_HS, y_test_HS])
	print("Kolmogorov-Smirnov (KS) test for ICU Stay",  test_ICU)
	print("Kolmogorov-Smirnov (KS) test for Hospital Stay", test_HS)
	print("Kolmogorov-Smirnov (KS) test for Heart Rate", test_HR)
	print("Kolmogorov-Smirnov (KS) test for Blood Pressure", test_IBP)
	print()
	print("Drift Distance for ICU Stay",wasserstein_distance(y_train_ICU, y_test_ICU))
	print("Drift Distance for Hospital Stay",wasserstein_distance(y_train_HS, y_test_HS))
	print("Drift Distance for Heart Rate",wasserstein_distance(y_train_HR, y_test_HR))
	print("Drift Distance for Blood Pressure",wasserstein_distance(y_train_IBP, y_test_IBP))
	#print("KL Divergence for ICU_Stay",res_ICU.statistic, res_ICU.pvalue,res_ICU.critical_values)
	#print("KL Divergence for Hospital_Stay",res_HS.statistic, res_HS.pvalue,res_HS.critical_values)
	print()
	#print("Jenen Shannon(JS) for ICU_Stay",js(y_testICU[0:70], y_trainICU[0:70]))
	#print("Jenen shannon (JS) for Hospital_Stay",js(y_testHS[0:205],y_trainHS[0:205]))
	print()
	
	print("CV for ICU",stats.cramervonmises_2samp(y_train_ICU,y_test_ICU))
	print("CV for hs",stats.cramervonmises_2samp(y_train_HS,y_test_HS))
	print("CV for hr", stats.cramervonmises_2samp(y_train_HR,y_test_HR))
	print("cv for ibp",stats.cramervonmises_2samp(y_train_IBP,y_test_IBP))
	print()
	
	"""print("icu",statsmodels.sandbox.stats.runs.runstest_2samp(y_train_ICU,y_test_ICU))
	print("hs",statsmodels.sandbox.stats.runs.runstest_2samp(y_train_HS,y_test_HS))
	print("hr",statsmodels.sandbox.stats.runs.runstest_2samp(y_train_HR,y_test_HR))
	print("ibp",statsmodels.sandbox.stats.runs.runstest_2samp(y_train_IBP,y_test_IBP))"""
	
	p = mannwhitneyu(y_train_HS, y_test_HS)
	print("mann  for hospital stay",p)
	p = mannwhitneyu(y_train_ICU, y_test_ICU)
	print("mann tst for ICU stay",p)
	p = mannwhitneyu(y_train_HR, y_test_HR)
	print("mann  for HR",p)
	p = mannwhitneyu(y_train_IBP, y_test_IBP)
	print("mann tst for IBP",p)

	"""
	np.random.seed(12345)
	ph = PageHinkley(threshold=10,min_instances=10)
	# Update drift detector and verify if change is detected
	data_stream=[]
	a = y_train_HS
	b = y_test_HS
	data_stream = np.concatenate((a,b))
	for i, val in enumerate(data_stream):
		in_drift, in_warning = ph.update(val)
		if in_drift:
			print(f"Change detected at index {i} for column with input value: {val}")
	
	adwin = ADWIN()
	data_stream=[]
	a = y_train_HS
	b = y_test_HS
	data_stream = np.concatenate((a,b))
	print(len(data_stream))
	# Adding stream elements to ADWIN and verifying if drift occurred
	for i in range(len(data_stream)):
		adwin.add_element(data_stream[i])
		print(data_stream[i])
		if adwin.detected_change():
			print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i) +'for column:ICU' )
		else:
			print("no change detected")
	
	"""

	

	
    
			
		
