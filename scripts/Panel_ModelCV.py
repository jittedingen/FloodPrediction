#### FUNCTION TO PERFORM PANEL LEVEL MODEL
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from tensorflow.keras import initializers
from keras.callbacks import EarlyStopping
from keras.layers import Dropout  # regularization
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Evaluate the predictions
def evaluate(y_test, pred):
    score = f1_score(y_test, pred)
    misses = sum((y_test == 1) & (pred == 0))
    hits = sum((y_test == 1) & (pred == 1))
    false_alarms = sum((y_test == 0) & (pred == 1))
    corr_neg = sum((y_test == 0) & (pred == 0))

    if (hits + false_alarms + misses) == 0:
        HFM = 0.0001
    else:
        HFM = hits + false_alarms + misses

    csi = hits / (HFM)

    return (score, csi, hits, misses, false_alarms, corr_neg)

# Split the data into folds depending on the blocks that were constructed to tackle the imbalance
def DataSplit(df):
    total_list = [1, 2, 3, 4, 5]*(math.ceil(len(df['block'].unique())/5))
    if (len(df['block'].unique()) % 5) != 0:
        total_list = total_list[0:-(5-(len(df['block'].unique()) % 5))]

    block_info = pd.DataFrame(df['block'].unique())
    block_info['fold'] = total_list
    block_info.columns = ['block', 'fold']
    return block_info

### Define a function that transforms the data into the right format for LSTM
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = np.array(sequences.iloc[i:end_ix,:].drop(columns='flood')), np.array(sequences.iloc[end_ix-1]['flood'])
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function which fits an LSTM model to the data and makes predictions
def panel_model(data_panel, NB_info, window, n_ahead, timesteps, activation, rec_activation, hidden_nodes, ep, drop_rate):
    from numpy.random import seed
    seed(1)
    tf.random.set_seed(2)

    ## Step 0) Normalize the vegetation, elevation and water surface data over the whole data set
    to_normalize = ['EVI_mean', 'perc_water', 'avg_slope']
    scaler = RobustScaler(with_centering=True)
    data_panel[to_normalize] = scaler.fit_transform(data_panel[to_normalize])

    ## Step 1) Determine the best performing station for each district & adjust the data accordingly
    my_local_path = str(Path(os.getcwd()))
    station_results = pd.read_csv(my_local_path + '/Results/LR/COEFF_LRw' + str(window) + '_ahead'+str(n_ahead) + 'NoWT.csv').filter(regex='dis_station|district')
    best_stations = station_results.set_index('district').idxmin(axis=1)
    best_stations.columns = ['district', 'best_station']

    NB_adjusted = pd.DataFrame()
    panel_df = pd.DataFrame()
    for district in NB_info['district'].unique():
        NB_d = NB_info[NB_info['district'] == district]
        df_district = data_panel[data_panel['district'] == district]

        if district in best_stations.reset_index()['district'].unique():
            station = best_stations.loc[district]
            NB_d = NB_d.filter(regex='district|NB_flood|time|' + station)
            station_cols = NB_d.filter(regex=station).columns
            for col in station_cols:
                col_name = col.replace(station, "")
                col_name = col_name.replace('NB_max_', "")
                NB_d = NB_d.rename(columns={col: "NB_best_dis_" + col_name})
            drop_cols = NB_d.filter(regex='dis_station').columns
            NB_d = NB_d.drop(columns=drop_cols)

            # Also update the district level info
            for col in df_district.filter(regex=station).columns:
                col_name = col.replace(station, "")
                df_district['best_dis' + col_name] = df_district[col]
            drop_cols = df_district.filter(regex='dis_station').columns
            df_district = df_district.drop(columns=drop_cols)

        # When no model is constructed for this district since the number of floods is low
        # Calculate the maximum normalized river discharge
        else:
            dis_cols = NB_d.filter(regex='dis_station').columns
            dis_cols_panel = df_district.filter(regex='dis_station').columns

            # check whether these columns contain details/smooth columns as well
            # if yes, then calculate the maximum normalized value per detail/smooth variable
            detail_cols = NB_d[dis_cols].filter(regex='details').columns
            if len(detail_cols) > 0:

                #first focus on details
                for i in range(1, 8, 1):
                    detail_df = NB_d[dis_cols].filter(regex='details_'+str(i))

                    # if this detail does not exist anymore, then we know the decomposition level
                    if detail_df.shape[1] == 0:
                        J = i-1
                        break

                    # if it still exists, get the maximum value over all detail cols in detail_df
                    else:
                        NB_d['NB_best_dis_details_'+ str(i)] = detail_df.max(axis=1)
                        df_district['best_dis_details_' + str(i)] = df_district[ dis_cols_panel].filter(regex='details_'+str(i)).max(axis=1)

                #Now that we know the decomposition level, apply the same strategy for the smooth
                NB_d['NB_best_dis_smooths_' + str(J)] = NB_d[dis_cols].filter(regex='smooths_' + str(J)).max(axis=1)
                df_district['best_dis_smooths_' + str(J)] = df_district[ dis_cols_panel].filter(regex='smooths_' + str(J)).max(axis=1)
                NB_d = NB_d.drop(columns=dis_cols)
                df_district = df_district.drop(columns =  dis_cols_panel)

            # if WT is not applied, do
            else:
                NB_d['NB_best_dis'] = NB_d[dis_cols].max(axis=1)
                df_district['best_dis'] = df_district[dis_cols_panel].max(axis=1)
                NB_d = NB_d.drop(columns = dis_cols)
                df_district = df_district.drop(columns = dis_cols_panel)

        NB_adjusted = pd.concat([NB_adjusted, NB_d], axis=0, ignore_index=True)
        panel_df = pd.concat([panel_df, df_district], axis=0, ignore_index=True)

    NB_info = NB_adjusted
    data_panel = panel_df

    ## Step 1) Attach NB info
    location = pd.read_csv(my_local_path + '/SpatialMatrix.csv')
    location['District'] = location['District'].str.lower()

    df = pd.DataFrame()
    for curr_district in data_panel['district'].unique():
        neighb = location[location['District'] == curr_district]
        neighb = neighb[neighb.columns[~neighb.isnull().all()]].drop(['District'], axis = 1).columns.str.lower()
        df_neighb = NB_info[NB_info['district'].isin(neighb)]

        # Now that we have all information about the neighbours, take the maximum over each columns
        df_district = data_panel[data_panel['district'] == curr_district]
        for col in df_neighb.filter(regex='NB_').columns:
            neighb_part = df_neighb[['time',col]]
            neighb_part = neighb_part.groupby('time').max().reset_index()

            # Attach to panel data
            df_district = pd.merge(df_district, neighb_part, how = 'left', on = 'time')

        # Now add to new dataframe
        df = pd.concat([df, df_district], axis=0, ignore_index=True)

    data_panel = df

    ## Step 2) Introduce district dummies
    district_dum = pd.get_dummies((data_panel['district'].astype('category')))
    data_panel = pd.concat([data_panel, district_dum], axis=1).drop(columns=['kwale'])

    ###### Remove unnecessary columns & pad columns with NaN with zeros (caused by different number of stations)
    data_panel = data_panel.drop(columns = ['time', 'Country'])
    data_panel = data_panel.fillna(0)

    ## Move the flood column to the end, such that the index of the block column does not change later in the script
    flood_col = data_panel['flood']
    data_panel = data_panel.drop(columns = 'flood')
    data_panel['flood'] = flood_col
    ix_Gcol = data_panel.columns.get_loc('block') #get index from column containing the block specification
    ix_district = data_panel.columns.get_loc('district') #get index from column containing the district specification

    ###### INTRODUCE CROSS-VALIDATION TO GET A MORE ROBUST RESULT ######
    block_info = DataSplit(data_panel)
    data_panel = pd.merge(data_panel, block_info, how='left', on='block')
    performance_fold = pd.DataFrame()
    district_info = pd.DataFrame()
    for fold in [1, 2, 3, 4, 5]:
        test = data_panel[data_panel['fold'] == fold]
        train = data_panel[~data_panel.index.isin(test.index)]
        testblocks = test['block'].unique()

        ## Step 2) Prepare the data for the LSTM
        ## For each block create samples in a 3D format
        ## Concatenate all these samples to each other
        ## Immediately divide in train and test

        X_train =np.zeros((1, timesteps, data_panel.shape[1]-1))
        y_train = np.zeros(1)
        X_test = np.zeros((1, timesteps, data_panel.shape[1]-1))
        y_test = np.zeros(1)

        for block in data_panel['block'].unique():
            df_b = data_panel[data_panel['block'] == block]

            if block in testblocks:
                X_testblock, y_testblock = split_sequences(df_b, timesteps)
                X_test = np.vstack((X_test, X_testblock))
                y_test = np.concatenate((y_test, y_testblock), axis = 0)

            else:
                X_trainblock, y_trainblock = split_sequences(df_b, timesteps)
                X_train = np.vstack((X_train, X_trainblock))
                y_train = np.concatenate((y_train, y_trainblock), axis = 0)

        ## Remove the first instances of X_train/test and y_train/test
        X_train = np.delete(X_train, (0), axis=0)
        X_test = np.delete(X_test, (0), axis=0)
        y_train = np.delete(y_train, (0)).reshape((len(y_train)-1, 1))
        y_test = np.delete(y_test, (0)).reshape((len(y_test)-1, 1))

        ## Step 3) Define balanced class_weights
        floods = y_train.sum() + y_test.sum()
        no_floods = len(y_train) + len(y_test) - floods
        class_weights = {0: ((len(y_train) + len(y_test)) / (2 * no_floods)),
                         1: ((len(y_train) + len(y_test)) / (2 * floods))}

        ## Step 4) Define components Group K Fold cross-validation
        groups = X_train[:,:,ix_Gcol][:,0]
        test_districts = X_test[:,:,ix_district][:,0]

        X_train = np.delete(X_train, X_train.shape[2]-1, axis = 2) # remove the fold column
        X_test = np.delete(X_test, X_test.shape[2]-1, axis = 2) # remove the fold column
        X_train = np.delete(X_train, ix_Gcol, axis = 2) # remove the block column
        X_test = np.delete(X_test, ix_Gcol, axis = 2) # remove the block column
        X_train = np.delete(X_train, ix_district, axis = 2).astype(np.float64) # remove the district column
        X_test = np.delete(X_test, ix_district, axis = 2).astype(np.float64) # remove the district column

        G_kfold = GroupKFold(n_splits=5)

        rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        #batch_size = [1024, X_train.shape[0]] just use X_train.shape[0] since our data is not super large so minibatch is not necessary
        res_GKFold = pd.DataFrame(1000, index=np.arange(len(rates)), columns = ['learning_rate', 'avg_loss', 'avg_acc'])
        loss_per_fold = []
        acc_per_fold = []
        i = 0
        """ This comment below calculates the best initial learning rate. However, we continue with 0.001
        for rate in rates:
            for train, val in G_kfold.split(X_train, y_train, groups = groups):
                try:
                ## Step 4) Initialize the LSTM & train the model
                    model = Sequential()
                    model.add(LSTM(units=hidden_nodes, activation=activation, recurrent_activation=rec_activation,
                               input_shape=(timesteps, X_train.shape[2])))
                    model.add(Dropout(drop_rate))
                    model.add(Dense(1)) #1 node here vs y is 1 dimension OF dense is 2 here and y is 2 dimensions
                    model.add(Activation('sigmoid'))

                    opt = Adam(learning_rate=rate)
                    model.compile(optimizer=opt, loss='binary_crossentropy', metrics = 'binary_crossentropy')
                    model.fit(X_train[train], y_train[train], epochs=ep, verbose=0, class_weight=class_weights)
                except:
                    print("error occurred")

                ## Step 5) Make predictions for this
                scores = model.evaluate(X_train[val], y_train[val], verbose=0)
                loss_per_fold.append(scores[0])
                acc_per_fold.append(scores[1])

            res_GKFold['avg_acc'].iloc[i] = acc_per_fold.mean()
            res_GKFold['avg_loss'].iloc[i] = loss_per_fold.mean()
            res_GKFold['learning_rate'].iloc[i] = rate
            i = i + 1

        ## Find the most optimal learning rate
        best_rate = res_GKFold['learning_rate'].iloc[res_GKFold['avg_loss'].argmin()] #should this be argmax or argmin?
        """
        best_rate = 0.001

        ## Step 5) Make predictions with the best learning rate that is found
        ## And calculate the results for 10 different weight initializations
        ## Then save all of the results and initializations, and retrieve the one with the best results
        inits = pd.DataFrame()
        for seed in range(1, 11,1):
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30, min_delta=0.001)
            model = Sequential()
            model.add(LSTM(units=hidden_nodes, activation=activation, recurrent_activation=rec_activation,
                           input_shape=(timesteps, X_train.shape[2]), kernel_initializer=initializers.GlorotUniform(seed=seed*5),
                           bias_initializer=initializers.Zeros(), stateful=False))
            model.add(Dropout(drop_rate))
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            opt = Adam(learning_rate=best_rate)
            model.compile(optimizer=opt, loss='binary_crossentropy')
            model.fit(X_train, y_train, epochs=ep, validation_split=0.3, verbose=1, class_weight=class_weights, batch_size=X_train.shape[0], shuffle = True, callbacks = [es]) #32: mini-batch GD
            pred = model.predict_classes(X_test)

            #### plot accuracy vs epochs
            # list all data in history
            """ Commented to prevent the popup of figures
            print(history.history.keys())
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            """

            ## Step 6) Evaluate & save the results for this seed
            score, csi, hits, misses, false_alarms, corr_neg = evaluate(y_test[:,0], pred[:,0])
            inits = inits.append([[seed, score, csi, hits, misses, false_alarms, corr_neg]])

        # Now we have the results for all different weight initializations --> check which one was best
        inits.columns = ['seed', 'f1', 'csi', 'hits', 'misses', 'false_alarms', 'corr_neg']
        inits = inits.reset_index(drop=True)
        best_init = inits.loc[inits['f1'].argmax()]

        # Get information on how each district performs separately
        district_df = pd.concat([pd.DataFrame(y_test), pd.DataFrame(pred), pd.DataFrame(test_districts)], axis = 1)
        district_df.columns = ['y_test', 'pred', 'district']
        for d in district_df['district'].unique():
            df_d = district_df[district_df['district'] == d]
            score, csi, hits, misses, false_alarms, corr_neg = evaluate(df_d['y_test'], df_d['pred'])
            district_info = district_info.append([[fold, d, score, csi, hits, misses, false_alarms, corr_neg]], ignore_index=True)


        performance_fold = performance_fold.append([[fold, best_rate, pred, y_test, best_init['f1'], best_init['csi'],
                                                     best_init['hits'], best_init['misses'], best_init['false_alarms'],
                                                     best_init['corr_neg']]], ignore_index=True)

    # Now we have the results for all folds, take the average over all
    results = pd.DataFrame(performance_fold.mean()).T
    results.columns = ['fold', 'best_rate', 'f1', 'csi', 'hits', 'misses', 'false_alarms', 'corr_neg']
    results_SE = pd.DataFrame(performance_fold.std()).T
    results_SE.columns = ['fold', 'best_rate', 'f1', 'csi', 'hits', 'misses', 'false_alarms', 'corr_neg']

    # Also get the average results per district
    district_info.columns = ['fold', 'district', 'f1', 'csi', 'hits', 'misses', 'false_alarms', 'corr_neg']
    avg_district_info = district_info.groupby(['district']).mean().reset_index()
    SE_district_info = district_info.groupby(['district']).std().reset_index()


    return results, results_SE, avg_district_info, SE_district_info
