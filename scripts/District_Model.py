#### FUNCTION TO PERFORM DISTRICT LEVEL MODEL
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import math


# Split the data into folds depending on the blocks that were constructed to tackle the imbalance
def DataSplit(df):
    total_list = [1, 2, 3, 4, 5]*(math.ceil(len(df['block'].unique())/5))
    if (len(df['block'].unique()) % 5) != 0:
        total_list = total_list[0:-(5-(len(df['block'].unique()) % 5))]

    block_info = pd.DataFrame(df['block'].unique())
    block_info['fold'] = total_list
    block_info.columns = ['block', 'fold']
    return block_info

def district_model(d, df, model, district_level):
    #### Insert extra 5-fold cross-validation to provide a better overall performance
    block_info = DataSplit(df)
    df = pd.merge(df, block_info, how='left', on='block')

    performance_fold = pd.DataFrame()
    coeffs_folds = pd.DataFrame()

    # For each fold, perform the following operations
    for fold in [1, 2, 3, 4, 5]:
        test = df[df['fold'] == fold]
        train = df[~df.index.isin(test.index)]
        groups = train['block']

        if district_level == True:
            #remove columns that are not used for modelling
            train = train.drop(columns = ['time', 'district', 'Country', 'block', 'fold'])
            test = test.drop(columns = ['time', 'district', 'Country', 'block', 'fold'])

            y_train = train['flood']
            X_train = train.drop(['flood'], axis=1)
            y_test = test['flood']
            X_test = test.drop(['flood'], axis=1)

            print('Number of floods in train (size)-test (size) for district ' + d + ' equals ' +
                  str(len(train[train['flood']==1])) + '(' + str(len(train)) + ')' + '-' +
                  str(len(test[test['flood']==1])) + '(' + str(len(test)) + ')')


            ## Perform Group K-fold CV
            train_blocks = len(groups.unique())

            if train_blocks >= 5: #do group 5-fold CV, otherwise do train_blocks-fold CV
                folds_cv = 5
            elif (train_blocks > 2) and (train_blocks < 5):
                folds_cv = train_blocks
            else:
                print('Too little flood events to construct a model for ' + d)
                folds_cv = 0
                pred = 1000
                prob = 1000
                y_test = pd.Series([1000, 1000])
                best_params = 'none'
                score = 1000
                blocks_intest = np.array([1000, 1000, 1000])

                hits = 'none'
                misses='none'
                false_alarms = 'none'
                corr_neg = 'none'
                pod = 'none'
                pofd = 'none'
                csi = 'none'
                far = 'none'
                coeffs = []


            if folds_cv > 2:
                G_kfold = GroupKFold(n_splits=folds_cv)

                if model == 'LR':
                    # LOGISTIC REGRESSION
                    logreg = LogisticRegression(class_weight='balanced', max_iter=1000)  # to adjust for the imbalance
                    grid = {"C":[0, 0.01, 0.05, 0.1, 0.5, 1, 1.25, 1.5], "penalty":["l2"]}# l2 ridge, l1 lasso
                    logreg_cv = GridSearchCV(logreg, param_grid=grid, cv=G_kfold, scoring='f1')
                    logreg_cv.fit(X_train, y_train, groups=groups)

                    print('Results for ' + d)
                    print("tuned hyperparameters :(best parameters) ", logreg_cv.best_params_)

                    logreg_final = LogisticRegression(class_weight='balanced', C=logreg_cv.best_params_['C'], penalty=logreg_cv.best_params_['penalty'],
                                                          max_iter=1000).fit(X_train, y_train)
                    pred = logreg_final.predict(X_test)
                    prob = logreg_final.predict_proba(X_test)
                    best_params = logreg_cv.best_params_

                    # Get ranking from most important variables
                    coeffs = pd.DataFrame(abs(logreg_final.coef_))
                    coeffs.columns = X_test.columns
                    coeffs = coeffs.T
                    coeffs['rank'] = coeffs[0].rank(ascending = False)
                    coeffs['var'] = coeffs.index
                    coeffs = coeffs.drop(columns = 0)

                    # Add the ranking of this fold to the ranking of the next folds to keep track of the overall important variables
                    if coeffs_folds.empty:
                        coeffs_folds = coeffs
                    else:
                        coeffs_folds = pd.merge(coeffs_folds, coeffs, on = 'var')

                elif model == 'SVC':
                    # SUPPORT VECTOR MACHINE
                    svm = SVC(probability=True, class_weight='balanced', max_iter=2000)
                    grid_svm = {"C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1],
                                    "kernel": ["rbf", "poly", 'linear', 'sigmoid'],
                                    'degree': [2, 3], 'gamma': ['auto', 'scale']}
                    svm_cv = GridSearchCV(svm, param_grid=grid_svm, cv=G_kfold, scoring='f1')
                    svm_cv.fit(X_train, y_train, groups=groups)

                    print('SVM results for ' + d)
                    print("tuned hyperparameters :(best parameters) ", svm_cv.best_params_)

                    svm_final = SVC(probability=True, class_weight='balanced', C=svm_cv.best_params_['C'],
                                        degree=svm_cv.best_params_['degree'],
                                        max_iter=2000, kernel=svm_cv.best_params_['kernel'],
                                    gamma=svm_cv.best_params_['gamma']).fit(X_train, y_train)
                    pred = svm_final.predict(X_test)
                    prob = svm_final.predict_proba(X_test)
                    best_params = svm_cv.best_params_
                    coeffs_folds = []

                score = f1_score(y_test, pred)
                misses = sum((y_test == 1) & (pred == 0))
                hits = sum((y_test == 1) & (pred == 1))
                false_alarms = sum((y_test == 0) & (pred == 1))
                corr_neg = sum((y_test == 0) & (pred == 0))

                if (hits + misses) == 0:
                    HM = 0.0001
                else:
                    HM = hits + misses

                if (hits + false_alarms) == 0:
                    HF = 0.0001
                else:
                    HF = hits + false_alarms

                if (false_alarms + corr_neg) == 0:
                    FC = 0.0001
                else:
                    FC = false_alarms + corr_neg

                if (hits + false_alarms + misses) == 0:
                    HFM = 0.0001
                else:
                    HFM = hits + false_alarms + misses

                pod = hits / (HM)
                far = false_alarms / (HF)
                pofd = false_alarms / (FC)
                csi = hits / (HFM)

                performance_fold = performance_fold.append([[len(test[test['flood']==1]), score,
                                          hits/len(test), misses/len(test), false_alarms/len(test), corr_neg/len(test),
                                          pod, far, pofd, csi]],
                                         ignore_index=True)


    # Now we have the results for all folds, take the average over all
    results = pd.DataFrame(performance_fold.mean().append(performance_fold.std())).T
    results.columns = ['avg_floods', 'avg_f1', 'avg_hits', 'avg_misses', 'avg_FalseA', 'avg_corrneg', 'avg_pod',
                           'avg_far', 'avg_pofd', 'avg_csi', 'std_floods', 'std_f1', 'std_hits', 'std_misses', 'std_FalseA', 'std_corrneg', 'std_pod',
                           'std_far', 'std_pofd', 'std_csi']
    results['district'] = d

    return results