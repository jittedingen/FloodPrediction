#### This script performs the sign test
import pandas as pd
import os
import math
from pathlib import Path
my_local_path = str(Path(os.getcwd()))

# Define window length and how far to predict ahead
window = 7
ahead = 2

#Get all results for one interval ahead
my_local_path = str(Path(os.getcwd()))
SVM_WT = pd.read_csv(my_local_path + '/Results/SVC/performance_SVCw' + str(window) + '_ahead' + str(ahead)+'WT.csv')
SVM_NoWT = pd.read_csv(my_local_path + '/Results/SVC/performance_SVCw' + str(window) + '_ahead' + str(ahead)+'NoWT.csv')
LR_WT = pd.read_csv(my_local_path + '/Results/LR/performance_LRw'  + str(window) + '_ahead' + str(ahead)+'WT.csv')
LR_NoWT = pd.read_csv(my_local_path + '/Results/LR/performance_LRw' + str(window) + '_ahead' + str(ahead)+'NoWT.csv')
benchmark = pd.read_csv(my_local_path + '/Results/Benchmark/performance_BenchM_future' + str(window) + 'daysNEW.csv')
LSTM_NoWT = pd.read_csv(my_local_path + '/Results/LSTM/Rate_0.001/performance_LSTMw' + str(window) + '_ahead' + str(ahead)+'NoWT.csv')
LSTM_WT = pd.read_csv(my_local_path + '/Results/LSTM/Rate_0.001/performance_LSTMw' + str(window) + '_ahead' + str(ahead)+'WT.csv')

districts = LR_NoWT['district'].unique()
benchmark = benchmark[benchmark['district'].isin(districts)] #only keep those districts that are also used in other models for comparison
benchmark = benchmark.rename(columns={'f1': 'avg_f1', 'csi': 'avg_csi'})
benchmark = benchmark.reset_index(drop=True)
n = len(LR_NoWT)

### Test if LR is better than LR WT, SVM, SVM WT and the benchmark
# F1 score.
test_results = pd.DataFrame()

for model in [SVM_WT, SVM_NoWT, LR_WT, benchmark]:
    if model.equals(SVM_WT):
        model_str = 'SVM_WT'
    elif model.equals(SVM_NoWT):
        model_str = 'SVM_NoWT'
    elif model.equals(LR_WT):
        model_str = 'LR_WT'
    else:
        model_str = 'BM'

    M = sum(LR_NoWT['avg_f1'] > model['avg_f1'])
    Z = (M-(n/2))/(0.5*math.sqrt(n))

    # Reject H0: LR is better than other models if
    if (Z >= 1.645):
        res_f1 = "Significantly better"
    else:
        res_f1 = "Can't reject H0"

# CSI
    M_csi = sum(LR_NoWT['avg_csi'] > model['avg_csi'])
    Z_csi = (M_csi-(n/2))/(0.5*math.sqrt(n))

    # Reject H0: LR is better than other models if
    if (Z_csi >= 1.645):
        res_csi = "Significantly better"
    else:
        res_csi = "Can't reject H0"

    # Append these results to a dataframe
    test_results = test_results.append([['districtLevel', 'LR_NoWT>' + model_str, res_f1, res_csi]])

### Test if SVM is better than LR WT, SVM WT, LR WT and the benchmark
for model in [SVM_WT, LR_NoWT, LR_WT, benchmark]:
    if model.equals(SVM_WT):
        model_str = 'SVM_WT'
    elif model.equals(LR_NoWT):
        model_str = 'LR_NoWT'
    elif model.equals(LR_WT):
        model_str = 'LR_WT'
    else:
        model_str = 'BM'

    M = sum(SVM_NoWT['avg_f1'] > model['avg_f1'])
    Z = (M-(n/2))/(0.5*math.sqrt(n))

    # Reject H0: LR is better than other models if
    if (Z >= 1.645):
        res_f1 = "Significantly better"
    else:
        res_f1 = "Can't reject H0"

# CSI
    M_csi = sum(SVM_NoWT['avg_csi'] > model['avg_csi'])
    Z_csi = (M_csi-(n/2))/(0.5*math.sqrt(n))

    # Reject H0: LR is better than other models if
    if (Z_csi >= 1.645):
        res_csi = "Significantly better"
    else:
        res_csi = "Can't reject H0"

    # Append these results to a dataframe
    test_results = test_results.append([['districtLevel','SVM_NoWT>' + model_str, res_f1, res_csi]])

### Test if LR WT is better than LR, SVM, SVM WT and the benchmark
for model in [SVM_WT, SVM_NoWT, LR_NoWT, benchmark]:
    if model.equals(SVM_WT):
        model_str = 'SVM_WT'
    elif model.equals(SVM_NoWT):
        model_str = 'SVM_NoWT'
    elif model.equals(LR_NoWT):
        model_str = 'LR_NoWT'
    else:
        model_str = 'BM'

    M = sum(LR_WT['avg_f1'] > model['avg_f1'])
    Z = (M-(n/2))/(0.5*math.sqrt(n))

    # Reject H0: LR is better than other models if
    if (Z >= 1.645):
        res_f1 = "Significantly better"
    else:
        res_f1 = "Can't reject H0"

# CSI
    M_csi = sum(LR_WT['avg_csi'] > model['avg_csi'])
    Z_csi = (M_csi-(n/2))/(0.5*math.sqrt(n))

    # Reject H0: LR is better than other models if
    if (Z_csi >= 1.645):
        res_csi = "Significantly better"
    else:
        res_csi = "Can't reject H0"

    # Append these results to a dataframe
    test_results = test_results.append([['districtLevel','LR_WT>' + model_str, res_f1, res_csi]])

### Test if SVM WT is better than LR, LR WT, SVM and the benchmark
for model in [LR_NoWT, SVM_NoWT, LR_WT, benchmark]:
    if model.equals(LR_NoWT):
        model_str = 'LR_NoWT'
    elif model.equals(SVM_NoWT):
        model_str = 'SVM_NoWT'
    elif model.equals(LR_WT):
        model_str = 'LR_WT'
    else:
        model_str = 'BM'

    M = sum(SVM_WT['avg_f1'] > model['avg_f1'])
    Z = (M-(n/2))/(0.5*math.sqrt(n))

    # Reject H0: LR is better than other models if
    if (Z >= 1.645):
        res_f1 = "Significantly better"
    else:
        res_f1 = "Can't reject H0"

# CSI
    M_csi = sum(SVM_WT['avg_csi'] > model['avg_csi'])
    Z_csi = (M_csi-(n/2))/(0.5*math.sqrt(n))

    # Reject H0: LR is better than other models if
    if (Z_csi >= 1.645):
        res_csi = "Significantly better"
    else:
        res_csi = "Can't reject H0"

    # Append these results to a dataframe
    test_results = test_results.append([['districtLevel','SVM_WT>' + model_str, res_f1, res_csi]])

test_results.columns = ['Level', 'test', 'f1', 'csi']
test_results.to_csv(str(Path(os.getcwd())) + "/Results/Sign_Test_w" + str(window) + "_ahead" + str(ahead) + ".csv")
test_results.head()