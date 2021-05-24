############ Wavelet Transform Visulatization #############
##############################################
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pywt
import matplotlib.pyplot as plt
import datetime

my_local_path = str(Path(os.getcwd()))
df = pd.read_csv(my_local_path + '/prep_WT_df.csv')

#COUNT NUMBER OF FLOODS PER DISTRICT
number_floods = pd.DataFrame([(district, df[df['district'] == district]['flood'].sum()) for district in df['district'].unique()], columns = ['district', 'total_floods']).sort_values(by='total_floods')

#1) ONLY VISUALIZATION
visualize_WT = True
if visualize_WT == True:
    wavelet = 'db4'
    #L = 2
    #J_0 = np.floor(log((8192/(L-1)) + 1, 2))
    J = 5

    district = 'butaleja'
    df_district = df[df['district'] == district] #test for one district
    df_district['flood'] = df_district['flood'].astype('category')
    df_district = df_district.dropna(axis=1, how='all')
    wavelet_vars = df_district.columns[(df_district.dtypes.values == np.dtype('float64'))].tolist()



    for var in wavelet_vars:

        # Since this code does not take data with any lengths we have to pad the data to be a multiple of 2^j
        signal = df_district[var]
        N = len(signal)
        extras = 8192 - N #8192 = 2^13 --> highest integer which is a multiple of 2^j
        pad_signal = np.pad(signal, (int(extras/2),int(extras/2)), 'mean') #insert mean both at the end and beginning to overcome boundary situation

        coeffs = pywt.swt(pad_signal, wavelet = wavelet, level = J, trim_approx=False, norm = True)

        # Plot results of WT
        fig_res, axs = plt.subplots(J+1,1)
        fig_aprox, axs_a = plt.subplots(J+1, 1)
        for j in range(1,J+1):
            level_j = coeffs[j-1]
            #if j == 1:
             #   axs[j-1].plot(level_j[0][int(extras/2):(8192-int(extras/2))]) #plot approximation
              #  axs[j-1].set_title('Smooths Level ' + str(J+1-j))
            # Add /2 to get a clearer picture of the transform

            #axs[j].plot(level_j[1][int(extras/2):(8192-int(extras/2))])  #plot details
            #axs[j].set_title('Details Level ' + str(J + 1 - j))
            axs[j-1].plot(level_j[1][int(extras/2):(8192-int(extras/2))//8])  #plot details
            axs[j-1].set_title('Details Level ' + str(J + 1 - j))

            axs_a[j-1].plot(level_j[0][int(extras/2):(8192-int(extras/2))//8]) #plot approximation
            axs_a[j-1].set_title('Smooth Level ' + str(J+1-j))
            if j == J:
                axs_a[j].plot(pad_signal[int(extras/2):(8192-int(extras/2))//8]) #plot original signal
                axs_a[j].set_title('Original Signal')

                axs[j].plot(pad_signal[int(extras/2):(8192-int(extras/2))//8]) #plot original signal
                axs[j].set_title('Original Signal')

        fig, ax = plt.subplots(1,1)
        ax.plot(pad_signal)
        fig.savefig(my_local_path + '/General Plots/Wavelet Transform/signal_%s_%s.png' % (district, J))
        fig_res.savefig(my_local_path + '/General Plots/Wavelet Transform/%s/achtWT_res_%s_%s_%sWindow7.png' % (var, district, wavelet, J))
        fig_aprox.savefig(my_local_path + '/General Plots/Wavelet Transform/%s/achtSmooths_%s_%s_%sWindow7.png' % (var, district, wavelet, J))


plots = True
if plots == True:
    district = 'butaleja'
    df_district = df[df['district'] == district] #test for one district
    df_district['flood'] = df_district['flood'].astype('category')
    df_district = df_district.dropna(axis=1, how='all')
    wavelet_vars = df_district.columns[(df_district.dtypes.values == np.dtype('float64'))].tolist()

    for var in wavelet_vars:
        fig, ax = plt.subplots()  # figsize=(40,25))
        plt.plot(pd.to_datetime(df_district['time']), df_district[var], label= var, linewidth=1)
        #ax.plot(pd.to_datetime(df_district[df_district['flood'] == 1]['time']),
                #df_district[df_district['flood'] == 1][var], 'ko')

        plt.show()

print('hola')