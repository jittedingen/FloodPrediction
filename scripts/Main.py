########### THIS IS THE MAIN SCRIPT FOR THE FLOOD FORECASTING MODEL ###########

# Load necessary packages
import pandas as pd
import numpy as np
import os
import math
from pathlib import Path
from Interpolation import interpolate
from Extra_Features import create_features
from Wavelet_Transform import wavelet_transform
from Imbalance import make_balanced
from District_Model import district_model
from sklearn.preprocessing import RobustScaler
from Panel_ModelCV import panel_model
import warnings
warnings.filterwarnings("ignore")

### Set variables ###
WT = False #False indicates no Wavelet Transform is applied
district_level = False #False indicates that one model is made for all data
if district_level == False:
    model = 'LSTM'
else:
    model = 'LR' #LR stands for logistic regression. Another option here is: 'SVC' for support vector machine/classifier

agg_interval = True #If True predictions will be made at interval level (every 3, 5 or 7 days for example). Otherwise, at daily level

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
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

### Perform the following operations for predictions of a window length of 3, 5 and 7
# For example: when the window length equals 5 we predict whether a flood will occur in an interval of 5 days.
for window in [3, 5, 7]:
    # We only construct predictions for roughly two weeks in the future.
    # Depending on the window length, the maximum number of intervals we predict ahead differs
    # For example; when the window length equals 7. Two intervals already indicate two weeks (hence, max_ahead = 2)
    if window == 3:
        max_ahead = 5
    elif window == 5:
        max_ahead = 3
    else:
        max_ahead = 2

    # Then, for each window length make predictions of roughly two weeks ahead
    for n_ahead in range(1, max_ahead+1):

        ## Load the (downloaded) data
        ## We make use of the medium quality data (
        my_local_path = str(Path(os.getcwd()))
        df_total = pd.read_csv(my_local_path + '/df_total_mediumqual.csv')

        ## Basic data modifications
        # Remove Pallisa from the data set as it does not include soil and rainfall information
        df_total = df_total[df_total['district'] != 'pallisa']

        # Remove irrelevant columns & transform Country & season to categorical variables
        df_total = df_total.drop(columns=['mean'], axis=1)
        df_total['season'] = df_total['season'].astype('category')
        df_total['flood'] = df_total['flood'].astype('category')

        #################### Variable initialization ##########################
        all_districts = df_total['district'].unique()
        block = 1
        data_panel = pd.DataFrame()
        data_NB = pd.DataFrame()
        testblocks = []
        results_DL = pd.DataFrame()
        coeff_DL = pd.DataFrame()
        results_PCA = []
        prep_pca = pd.DataFrame()
        prep_wt = pd.DataFrame()
        #################################################################

        #### Prepare the data per district for the LSTM -OR- Construct a model per district ####
        #### This is determined by the district_level variable.
        for d in all_districts:
            ## Step 1) Interpolate missing values & retrieve district level data
            df_d = interpolate(d, df_total)

            ## Step 2) Create rolling window for flood variable to tackle delay flood event & reported impact
            ## So, if a flood is recorded on the 10th of May we denote that there was a flood on the 8th and 9th of May as well,
            ## to overcome the issue of potential delay in the recorded flood event.
            df_d['flood'] = df_d['flood'].rolling(3).max().shift(periods=-(2))
            df_d = df_d.dropna() #drop NaN created by this rolling window

            ## Step 3) Introduce extra features & remove irrelevant columns
            df_d, float_cols = create_features(df_d, district_level, agg_interval, window, n_ahead)

            ## Step 4) Resample over 'window' days
            if agg_interval == True:

                # Do a different operation for rainfall - sum instead of max like for the other variables
                # Take the sum of rainfall over each interval window
                rainfall_df = df_d[['time', 'rainfall']]
                rainfall_df = rainfall_df.set_index('time').resample(str(window)+"D").sum()
                df_d = df_d.drop(columns = 'rainfall')

                # shift the GloFAS cols if too far predicting ahead
                # GloFAS data is only available 7 days in the future
                glofas_cols = df_d.filter(regex='dis_station').columns
                if (window*n_ahead) > 7:
                    # shift the glofas data up
                    df_d[glofas_cols] = df_d[glofas_cols].shift(periods= - (7 % window))

                # now take the maximum over the remaining variables for each window + change flood to numerical
                df_d['flood'] = df_d['flood'].astype('int64')
                df_d = df_d.set_index('time').resample(str(window)+"D").max()

                # Attach transformed rainfall data
                df_d = pd.concat([df_d, rainfall_df], axis = 1)

                # Shift the GloFAS data to get information about GloFAS in the future at window level
                # Shift is necessary to teach the model how to predict ahead
                glofas_cols = df_d.filter(regex='dis_station').columns
                if window == 3:
                    if n_ahead >= 3:
                        df_d[glofas_cols] = df_d[glofas_cols].shift(periods= (n_ahead - 2))
                elif (window == 5) | (window ==7):
                    if n_ahead >= 2:
                        df_d[glofas_cols] = df_d[glofas_cols].shift(periods= (n_ahead - 1))

                # Shift the rainfall and soilM data to teach the model how to predict ahead
                rainsoil = df_d.filter(regex='rain|soil|NB_flood').columns
                df_d[rainsoil] = df_d[rainsoil].shift(periods = n_ahead)
                df_d = df_d.dropna() #drop rows with NA that are created by shifting the data

                df_d['time'] = df_d.index
                df_d = df_d.reset_index(drop=True)


            ## Change the types of some variables & prepare list of column names
            df_d['flood'] = df_d['flood'].astype('category')
            df_d['spring'] = df_d['spring'].astype('int64')
            df_d['summer'] = df_d['summer'].astype('int64')
            df_d['winter'] = df_d['winter'].astype('int64')
            float_cols = df_d.columns[(df_d.dtypes.values == np.dtype('float64'))].tolist()
            float_cols.insert(0, float_cols.pop())

            # Remove features that are added for panel level model for the wavelet transform
            # These variables are invariant over time and hence the wavelet can't be applied
            if district_level == False:
                float_cols.remove('EVI_mean')
                float_cols.remove('avg_slope')
                float_cols.remove('perc_water')
                float_cols.remove('NB_flood14days')

            ## Step 5) Wavelet Transform (Only perform if preferred)
            if WT == True:
                df_d = wavelet_transform(df_d, float_cols)

            ## Step 6) Normalize the data (WT results in very different scaled variables)
            ## First prepare the list of float columns again
            df_d['flood'] = df_d['flood'].astype('category')
            float_cols = df_d.columns[(df_d.dtypes.values == np.dtype('float64'))].tolist()
            if district_level == False:
                float_cols.remove('EVI_mean')
                float_cols.remove('avg_slope')
                float_cols.remove('perc_water')
                float_cols.remove('NB_flood14days')

            scaler = RobustScaler(with_centering=True)
            df_d[float_cols] = scaler.fit_transform(df_d[float_cols])

            ## Step 7) Tackle the imbalance in the data
            # NB stands for neighbouring districts or counties
            # Already save the information for the country level model (needed for later)
            if district_level == False:
                NB_cols = df_d.filter(regex='NB_|district|time').columns
                NB_info = df_d[NB_cols]
                NB_cols = df_d.filter(regex='NB_').columns
                df_d = df_d.drop(columns= NB_cols)

            df_d, block = make_balanced(d, df_d, block, agg_interval, window)


            ## Step 8) Modelling
            ######### DISTRICT LEVEL MODEL ##########
            if district_level == True:
                all_blocks = df_d['block'].unique()
                if len(all_blocks) < 5: # If less than 5 blocks can be made for a district, there are too little flood events to perform cross-validation
                    print('Too little blocks with flood events in district ' + d)
                    continue

                # Run the district level model & append all results into one dataframe
                else:
                    res_DL = district_model(d, df_d, model, district_level)
                    results_DL = results_DL.append(res_DL)

            # If we want to use the country level model, attach all data together (incl. the neighbour info)
            else:
                data_panel = pd.concat([data_panel, df_d], axis=0, ignore_index=True)
                data_NB = pd.concat([data_NB, NB_info], axis = 0, ignore_index=True)


        ## Save results if district level models are used ONLY
        if district_level == True:
            if WT == False:
                pd.DataFrame(results_DL).to_csv(str(Path(os.getcwd()))+"/Results/"+model+"/performance_"+model+"w"+str(window)+"_ahead"+str(n_ahead)+"noWT.csv")
            else:
                pd.DataFrame(results_DL).to_csv(str(Path(os.getcwd())) + "/Results/" + model + "/performance_" + model + "w" + str(window) + "_ahead" + str(n_ahead) + "WT.csv")



        ######## COUNTRY LEVEL MODEL #########
        if district_level == False:

            # Variable setting
            timesteps = math.ceil(45 / window) #set the number of timesteps to the number of intervals that span 1.5 months
            activation = 'tanh'
            rec_activation = 'sigmoid'
            ep = 500
            hidden_nodes = 32
            drop_rate = 0.2

            # Get an average performance + the performance per district with the country level model
            avg_perf, SE, district_avg, district_SE = panel_model(data_panel, data_NB, window, n_ahead, timesteps, activation, rec_activation, hidden_nodes, ep, drop_rate)

            # Save the results
            if WT == False:
                pd.DataFrame(district_avg).to_csv(str(Path(os.getcwd())) + "/Results/LSTM/performance_" + model + "PERDISTRICT_w" + str(window) + "_ahead" + str(n_ahead) + "NoWT.csv")
                pd.DataFrame(district_SE).to_csv(str(Path(os.getcwd())) + "/Results/LSTM/SE_" + model + "PERDISTRICTw" + str(window) + "_ahead" + str(n_ahead) + "NoWT.csv")
                pd.DataFrame(avg_perf).to_csv(str(Path(os.getcwd())) + "/Results/LSTM/performance_" + model + "w" + str(window) + "_ahead" + str(n_ahead) + "NoWT.csv")
                pd.DataFrame(SE).to_csv(str(Path(os.getcwd())) + "/Results/LSTM/SE_" + model + "w" + str(window) + "_ahead" + str(n_ahead) + "NoWT.csv")

            else:
                pd.DataFrame(district_avg).to_csv(str(Path(os.getcwd())) + "/Results/LSTM/performance_" + model + "PERDISTRICT_w" + str(window) + "_ahead" + str(n_ahead) + "WT.csv")
                pd.DataFrame(district_SE).to_csv(str(Path(os.getcwd())) + "/Results/LSTM/SE_" + model + "PERDISTRICTw" + str(window) + "_ahead" + str(n_ahead) + "WT.csv")
                pd.DataFrame(avg_perf).to_csv(str(Path(os.getcwd())) + "/Results/LSTM/performance_" + model + "w" + str(window) + "_ahead" + str(n_ahead) + "WT.csv")
                pd.DataFrame(SE).to_csv(str(Path(os.getcwd())) + "/Results/LSTM/SE_" + model + "w" + str(window) + "_ahead" + str(n_ahead) + "WT.csv")


print('done')