#### This script visualizes the results
import pandas as pd
import numpy as np
import os
from pathlib import Path
my_local_path = str(Path(os.getcwd()))
import seaborn as sns
import matplotlib.pyplot as plt

model = 'LR'
WT = False
score = 'csi' #f1 / csi / pod

# Get all results & examine which model performs best for each district
# Only do this for window 5 and n_ahead = 1
my_local_path = str(Path(os.getcwd()))
LSTM_NoWT = pd.read_csv(my_local_path + '/Results/LSTM/Rate_0.001/PERDISTRICT/performance_LSTMPERDISTRICT_w5_ahead2NoWT.csv')
LSTM_WT = pd.read_csv(my_local_path + '/Results/LSTM/Rate_0.001/PERDISTRICT/performance_LSTMPERDISTRICT_w5_ahead2WT.csv')
SVM_WT = pd.read_csv(my_local_path + '/Results/SVC/performance_SVCw5_ahead2WT.csv')
SVM_NoWT = pd.read_csv(my_local_path + '/Results/SVC/performance_SVCw5_ahead2NoWT.csv')
LR_WT = pd.read_csv(my_local_path + '/Results/LR/performance_LRw5_ahead2WT.csv')
LR_NoWT = pd.read_csv(my_local_path + '/Results/LR/performance_LRw5_ahead2NoWT.csv')
benchmark = pd.read_csv(my_local_path + '/Results/Benchmark/performance_BenchM_future5daysNEW.csv')

all_districts = LR_NoWT['district'].unique()
best_districtmodel = pd.DataFrame()
compare = True
if compare == True:
    for d in all_districts:
        best = 0
        res_d = LSTM_NoWT[LSTM_NoWT['district'] == d]
        if res_d['f1'].iloc[0] > best:
            best_model = 'LSTM_NoWT'
            best_f = res_d['f1'].iloc[0]
            best_csi = res_d['csi'].iloc[0]
            hits = res_d['hits'].iloc[0]
            misses = res_d['misses'].iloc[0]
            falseAl = res_d['false_alarms'].iloc[0]
            corrNeg = res_d['corr_neg'].iloc[0]
            total = hits + misses + falseAl + corrNeg
            best_hits = hits/total
            best_misses = misses/total
            best_falseAl = falseAl/total
            best_corrNeg = corrNeg/total
            best = best_f

        res_d = LSTM_WT[LSTM_WT['district'] == d]
        if res_d['f1'].iloc[0] > best:
            best_model = 'LSTM_WT'
            best_f = res_d['f1'].iloc[0]
            best_csi = res_d['csi'].iloc[0]
            hits = res_d['hits'].iloc[0]
            misses = res_d['misses'].iloc[0]
            falseAl = res_d['false_alarms'].iloc[0]
            corrNeg = res_d['corr_neg'].iloc[0]
            total = hits + misses + falseAl + corrNeg
            best_hits = hits/total
            best_misses = misses/total
            best_falseAl = falseAl/total
            best_corrNeg = corrNeg/total
            best = best_f

        res_d = SVM_WT[SVM_WT['district'] == d]
        if res_d['avg_f1'].iloc[0] > best:
            best_model = 'SVM_WT'
            best_f = res_d['avg_f1'].iloc[0]
            best_csi = res_d['avg_csi'].iloc[0]
            best_hits = res_d['avg_hits'].iloc[0]
            best_misses = res_d['avg_misses'].iloc[0]
            best_falseAl = res_d['avg_FalseA'].iloc[0]
            best_corrNeg = res_d['avg_corrneg'].iloc[0]
            best = best_f

        res_d = SVM_NoWT[SVM_NoWT['district'] == d]
        if res_d['avg_f1'].iloc[0] > best:
            best_model = 'SVM_NoWT'
            best_f = res_d['avg_f1'].iloc[0]
            best_csi = res_d['avg_csi'].iloc[0]
            best_hits = res_d['avg_hits'].iloc[0]
            best_misses = res_d['avg_misses'].iloc[0]
            best_falseAl = res_d['avg_FalseA'].iloc[0]
            best_corrNeg = res_d['avg_corrneg'].iloc[0]
            best = best_f

        res_d = LR_NoWT[LR_NoWT['district'] == d]
        if res_d['avg_f1'].iloc[0] > best:
            best_model = 'LR_NoWT'
            best_f = res_d['avg_f1'].iloc[0]
            best_csi = res_d['avg_csi'].iloc[0]
            best_hits = res_d['avg_hits'].iloc[0]
            best_misses = res_d['avg_misses'].iloc[0]
            best_falseAl = res_d['avg_FalseA'].iloc[0]
            best_corrNeg = res_d['avg_corrneg'].iloc[0]
            best = best_f

        res_d = LR_WT[LR_WT['district'] == d]
        if res_d['avg_f1'].iloc[0] > best:
            best_model = 'LR_WT'
            best_f = res_d['avg_f1'].iloc[0]
            best_csi = res_d['avg_csi'].iloc[0]
            best_hits = res_d['avg_hits'].iloc[0]
            best_misses = res_d['avg_misses'].iloc[0]
            best_falseAl = res_d['avg_FalseA'].iloc[0]
            best_corrNeg = res_d['avg_corrneg'].iloc[0]
            best = best_f
        """
        res_d = benchmark[benchmark['district'] == d]
        if res_d['f1'].iloc[0] > best:
            best_model = 'benchmark'
            best_f = res_d['f1'].iloc[0]
            best_csi = res_d['csi'].iloc[0]
            best_hits = res_d['hits'].iloc[0]
            best_misses = res_d['misses'].iloc[0]
            best_falseAl = res_d['false_alarms'].iloc[0]
            best_corrNeg = res_d['corr_neg'].iloc[0]
            best = best_f
        """
        best_districtmodel = best_districtmodel.append([[d, best_model, best_f, best_csi, best_hits, best_misses, best_falseAl, best_corrNeg]])

    best_districtmodel = pd.DataFrame(best_districtmodel)
    best_districtmodel.columns = ['district', 'best_model', 'best_f1', 'best_csi', 'best_hits', 'best_misses', 'best_falseAl', 'best_corrNeg']
    best_districtmodel.to_csv(my_local_path + '/Results/best_modelPerDistrictW5A2.csv')

floodcheck = True
if floodcheck == True:
    from Interpolation import interpolate

    my_local_path = str(Path(os.getcwd()))
    df_total = pd.read_csv(my_local_path + '/df_total_mediumqual.csv')

    check_res_total = []
    all_districts = df_total['district'].unique()
    for dis in all_districts:
        df_dis = interpolate(dis, df_total)
        n_floods = df_dis['flood'].sum()

        check_res = {'district':dis, 'n_floods':n_floods, 'country':df_dis['Country'].iloc[0]}
        check_res_total.append(check_res)

    res_floodcheck = pd.DataFrame(check_res_total)


#1) Benchmark 1.0 - max over 3 days ago, today and 3 days ahead
if model == 'benchM1':
    df = pd.read_csv(my_local_path + '/Results/Benchmark/performance_OriginalBenchM_NEW.csv')
    df_SE = pd.read_csv(my_local_path + '/Results/Benchmark/performance_OriginalBenchM_NEW_SE.csv')
    avg = df.mean()
    avg_SE = df_SE.mean()

#2) Benchmark 2.0 - max over the upcoming three days for different number of days ahead
elif model == 'benchM2':

    df_resavg = pd.DataFrame(index= [7, 5, 3], columns= [1, 3, 5])
    df_resmedian = pd.DataFrame(index= [7, 5, 3], columns= [1, 3, 5])

    # First attach the results from Kenya and Uganda for each n_ahead and window
    for ahead in [1, 3, 5]:
        for window in [3, 5, 7]:
            df_country = pd.DataFrame()
            for country in ['Kenya', 'Uganda']:
                if ahead == 1:
                    df_window = pd.read_csv(my_local_path + '/Results/Benchmark/n_ahead'+str(ahead)+'/performance_BenchM_'+country+'_future'+ str(window) +'days.csv')

                elif ahead==3:
                    if window == 7:
                        continue
                    else:
                        df_window = pd.read_csv(my_local_path + '/Results/Benchmark/n_ahead' + str(ahead) + '/performance_BenchM_' + country + '_future' + str(window) + 'days.csv')

                elif ahead == 5:
                    if window != 3:
                        continue
                    else:
                        df_window = pd.read_csv(my_local_path + '/Results/Benchmark/n_ahead' + str(ahead) + '/performance_BenchM_' + country + '_future' + str(window) + 'days.csv')

                df_country = pd.concat([df_country, df_window])

            #Calculate the average and median scores ---- CHANGE F1 / CSI
            if ((ahead==3) & (window==7)) | ((ahead==5) & (window != 3)):
                continue
            else:
                avg = df_country[score].mean()
                median = df_country[score].median()

                # Fill in results matrix
                df_resavg.loc[window][ahead] = avg
                df_resmedian.loc[window][ahead] = median

    #fig_avg = px.imshow(df_resavg)
    #fig_avg.write_image(my_local_path + '/Results/Benchmark/results_avg.png')
    #fig_median = px.imshow(df_resmedian)
    #fig_median.write_image(my_local_path + '/Results/Benchmark/results_median.png')

    # Change types of resulting matrices to float
    df_resavg = df_resavg.astype('float64')
    df_resmedian = df_resmedian.astype('float64')

    # Make the figures
    sns_avg = sns.heatmap(df_resavg, annot=True, vmin=0, vmax=1, cbar_kws={'label': 'CSI'})
    sns_avg.set_yticklabels(labels=df_resavg.index, rotation = 0)
    plt.ylabel("Interval Size")
    plt.xlabel("Number of days ahead")
    fig_avg = sns_avg.get_figure()
    fig_avg.savefig(my_local_path + '/Results/Benchmark/results_avgCSI.png')
    plt.clf()

    sns_median = sns.heatmap(df_resmedian, annot=True, vmin=0, vmax=1, cbar_kws={'label': 'CSI'})
    sns_median.set_yticklabels(labels=df_resmedian.index, rotation = 0)
    plt.ylabel("Interval Size (days)")
    plt.xlabel("Number of days ahead")
    fig_median = sns_median.get_figure()
    fig_median.savefig(my_local_path + '/Results/Benchmark/results_medianCSI.png')

elif model == 'benchM3': #without rolling average (so only forecasting 1 interval ahead)
    df = pd.read_csv(my_local_path + '/Results/Benchmark/performance_BenchM_future7daysNEW.csv')
    df_SE = pd.read_csv(my_local_path + '/Results/Benchmark/performance_BenchM_future7daysNEW_SE.csv')
    avg = df.mean()
    avg_SE = df_SE.mean()

elif model == "LR_OLD":
    if score == 'f1':
        score = 'score'
        label = 'F1 score'
    else:
        label = score

    df_resavg = pd.DataFrame(index= [7, 5, 3], columns= [1, 3, 5, 7, 9, 11, 13])
    df_resmedian = pd.DataFrame(index= [7, 5, 3], columns= [1, 3, 5, 7, 9, 11, 13])

    # First attach the results from Kenya and Uganda for each n_ahead and window
    for ahead in [1, 3, 5, 7, 9, 11, 13]:
        for window in [3, 5, 7]:
            if WT == True:
                df_part = pd.read_csv(my_local_path + '/Results/LR/performance_LRw' + str(window) + '_ahead' + str(ahead) + 'WT.csv')
                WT_indicator = 'WT'

            else:
                df_part = pd.read_csv(my_local_path + '/Results/LR/performance_LRw' + str(window) +'_ahead' + str(ahead) + 'NoWT.csv')
                WT_indicator = 'NoWT'

            # Remove scores that are equal to 1000 --> indicates that there were too little flood events to make a model
            df_part = df_part[~df_part[score].isin([1000, 'none'])]

            #Calculate the average and median scores ---- CHANGE F1 / CSI
            df_part[score] = df_part[score].astype('float64')
            avg = df_part[score].mean()
            median = df_part[score].median()

            # Fill in results matrix
            df_resavg.loc[window][ahead] = avg
            df_resmedian.loc[window][ahead] = median

    #fig_avg = px.imshow(df_resavg)
    #fig_avg.write_image(my_local_path + '/Results/Benchmark/results_avg.png')
    #fig_median = px.imshow(df_resmedian)
    #fig_median.write_image(my_local_path + '/Results/Benchmark/results_median.png')

    # Change types of resulting matrices to float
    df_resavg = df_resavg.astype('float64')
    df_resmedian = df_resmedian.astype('float64')

    # Make the figures
    sns_avg = sns.heatmap(df_resavg, annot=True, vmin=0, vmax=1, cbar_kws={'label': label})
    sns_avg.set_yticklabels(labels=df_resavg.index, rotation = 0)
    plt.ylabel("Interval Size")
    plt.xlabel("Number of days ahead")
    fig_avg = sns_avg.get_figure()
    fig_avg.savefig(my_local_path + '/Results/LR/results' + WT_indicator + '_avg' + score + '.png')
    plt.clf()

    sns_median = sns.heatmap(df_resmedian, annot=True, vmin=0, vmax=1, cbar_kws={'label': label})
    sns_median.set_yticklabels(labels=df_resmedian.index, rotation = 0)
    plt.ylabel("Interval Size (days)")
    plt.xlabel("Number of days ahead")
    fig_median = sns_median.get_figure()
    fig_median.savefig(my_local_path + '/Results/LR/results' + WT_indicator + '_median' + score + '.png')


elif model == "LR":
    title = 'LRw7_ahead1NoWT'
    df = pd.read_csv(my_local_path + '/Results/LR/performance_' + title +'.csv')
    avg = df.mean()

    # Make plot performance vs number of floods in district
    merged = pd.merge(res_floodcheck, df, on='district', how='inner').sort_values(by='n_floods', ascending=True)
    #plt.plot(merged['n_floods'], merged['avg_f1'])
    #plt.ylabel('Average F1')
    #plt.xlabel('Number of floods')
    #plt.show()
    #plt.savefig(my_local_path + '/Results/Performance_VS_Floods_'+ title +'.png')
    #plt.clf()

    # Make plot performance vs number of floods in district - with average performance over intervals of 5 floods
    intervals = []
    intervals_med = []
    for max in range(5, 60, 5):
        sub = merged[(merged['n_floods'] >= (max - 5)) & (merged['n_floods'] < max)]
        n_districts = len(sub)

        # take the average and median of the performance
        avg_int = sub['avg_f1'].mean()
        med_int = sub['avg_f1'].median()
        intervals.append([n_districts, avg_int, med_int, str(max-5)+ '-' + str(max-1)])


    intervals = pd.DataFrame(intervals)
    intervals.columns = ['n_districts', 'avg_f1', 'med_f1', 'interval']

    fig, ax1 = plt.subplots()
    B = 11
    bars = intervals['n_districts']

    ax1.plot(intervals['interval'], intervals['avg_f1'])
    ax1.set_ylabel('Average F1', fontsize = 13)
    ax1.set_xlabel('Number of floods', fontsize = 13)
    ax1.legend(['Average F1 Score'])

    #plot bars
    ax2 = ax1.twinx()

    # Set tick font size
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(13)

    ax2.bar(np.arange(B), bars, alpha = 0.3)
    ax2.set_ylabel('Number of districts in interval', fontsize= 13)
    ax2.legend(['Number of districts'])

    #plt.show()
    # Change the sizes
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.savefig(my_local_path + '/Results/IntervalAvgPerformance_VS_Floods_' + title +'.png', dpi=300)
    plt.clf()

elif model == 'SVC':
    title = 'SVCw7_ahead2WT'
    df = pd.read_csv(my_local_path + '/Results/SVC/performance_'+ title + '.csv')
    avg = df.mean()



print('done')

