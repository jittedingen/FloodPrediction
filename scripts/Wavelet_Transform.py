### FUNCTION THAT IS USED TO PERFORM THE WAVELET TRANSFORM
import pywt
import pandas as pd
import numpy as np

def wavelet_transform(df_d, float_cols):
    J=3 #maximum decomposition level
    wavelet = 'db4'
    wavelet_vars = float_cols

    for var in wavelet_vars:
        # Since this code does not take data with any lengths we have to pad the data to be a multiple of 2^j
        signal = df_d[var]
        N = len(signal)
        extras = 8192 - N  # 8192 = 2^13 --> highest integer which is a multiple of 2^j
        pad_signal = np.pad(signal, (int(np.floor(extras / 2)), int(np.ceil(extras / 2))), 'mean') # insert mean both at the end and beginning to overcome boundary situation
        coeffs = pywt.swt(pad_signal, wavelet=wavelet, level=J, trim_approx=False, norm=True)

        for j in range(1, J + 1):
            df_d[var + '_details_' + str(J + 1 - j)] = coeffs[j - 1][1][int(np.floor(extras / 2)):(
                        8192 - int(np.ceil(extras / 2)))]  # remove padded coeffs

            # Only keep the smooth of the Jth level
            if (j == 1):# or (j == 2): #keep the smooths for the Jth and J-1th level
                df_d[var + '_smooths_' + str(J + 1 - j)] = coeffs[j - 1][0][
                                                                       int(np.floor(extras / 2)):(8192 - int(np.ceil(extras / 2)))]

    df_d = df_d.drop(columns=wavelet_vars)
    return df_d