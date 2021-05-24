### FUNCTION IS USED TO TACKLE THE IMBALANCE IN THE DATA
import numpy as np
import pandas as pd
import math

def make_balanced(d, df_d, block_init, agg_interval, window):

    if agg_interval == True:
        # Cut the data 2 months before and after a flood
        interval_days = math.ceil(60/window)
        div = window
    else:
        # cut the data 3 weeks before a flood and 3 after
        interval_days = 21
        div = 1

    flood_points = df_d.index[df_d['flood'] == 1].tolist()
    n_floods = len(flood_points)
    cutted = pd.DataFrame()
    index_consecFloods = [idx for idx, val in enumerate(np.diff(flood_points)) if val < interval_days+1]
    block = block_init
    if len(index_consecFloods) > 0: # if there are consecutive floods make sure to put them in one block
        list = []
        for i in range(0, len(index_consecFloods)):
            if index_consecFloods[i] in list:
                continue

            list = [index_consecFloods[i]]
            j = i
            if j != len(index_consecFloods)-1: #make sure that the index of j+1 exists below
                while (index_consecFloods[j+1] - index_consecFloods[j] == 1): # ensure that we include more than 2 consecutive floods as well
                    list.append(index_consecFloods[j+1])
                    j = j + 1
                    if j == len(index_consecFloods)-1:
                        break


            minval = min(list)
            maxval = max(list)

            cutted_part = df_d.loc[flood_points[minval]-interval_days:flood_points[maxval+1]+interval_days] # get 3 weeks before first flood & 3 weeks after consecutive flood
            cutted_part['block'] = block
            cutted = pd.concat([cutted, cutted_part], axis = 0, ignore_index=True)
            block = block + 1

        # Remove the already "tackled" flood points from the flood points list
        # Such that the consecutive flood points are removed from this list and
        # below blocks are made for only non-consecutive floods
        list2 = [x+1 for x in index_consecFloods]
        to_remove = index_consecFloods + list2
        to_remove = set(to_remove) #remove duplicates
        for ele in sorted(to_remove, reverse=True):
            del flood_points[ele]

    for flood in flood_points:
        cutted_part = df_d.loc[flood-interval_days:flood+interval_days] #get data 3 weeks before and after
        cutted_part['block'] = block
        cutted = pd.concat([cutted, cutted_part], axis=0, ignore_index=True)
        block = block + 1

    #organize blocks such that the last ones are the most recent ones
    cutted = cutted.sort_values(by='time')

    #If some blocks overlap, recut the blocks again (this is possible for floods that are less than 6 weeks apart)
    #If overlap, then keep one week after flood of first block and keep two weeks before a flood of second block
    duplicates = cutted[cutted.duplicated(['time'], keep=False)]
    if len(duplicates) > 0:
        dup_blocks = duplicates['block'].unique()
        for dup in range(0,len(dup_blocks)):
            if dup != (len(dup_blocks)-1):
                dup_df = cutted[cutted['block'] == dup_blocks[dup]]
                dup_df_2 = cutted[cutted['block'] == dup_blocks[dup+1]]

                #drop the two blocks above containing duplicates from the cutted dataframe & update below
                cutted = cutted[~cutted['block'].isin(dup_blocks[[dup, dup+1]])]

                if dup_df['time'].min() < dup_df_2['time'].min(): #then first block comes before next one
                    latest_flood = dup_df.index[dup_df['flood'] == 1].tolist()[-1]  # get the index of the last flood
                    time_flood = dup_df.loc[latest_flood]['time']

                    newest_flood = dup_df_2.index[dup_df_2['flood'] == 1].tolist()[0]  # get the index of the newest flood
                    time_flood_new = dup_df_2.loc[newest_flood]['time']

                    diff_days = math.ceil((time_flood_new-time_flood).days/div)
                    if diff_days > interval_days*2:
                        cutted = pd.concat([cutted, dup_df, dup_df_2], axis=0)
                        continue
                    else:
                        extra_days = interval_days*2 - diff_days + 1
                        before_flood = int(extra_days / 3)  # give twice as much importance to data before a flood than after a flood
                        after_flood = extra_days - before_flood

                        dup_df = dup_df.iloc[:-after_flood]
                        dup_df_2 = dup_df_2.iloc[before_flood:]

                else: #then second block comes before first one
                    latest_flood = dup_df_2.index[dup_df_2['flood']==1].tolist()[-1]
                    time_flood = dup_df_2.loc[latest_flood]['time']

                    newest_flood = dup_df.index[dup_df['flood']==1].tolist()[0]
                    time_flood_new = dup_df.loc[newest_flood]['time']

                    diff_days = math.ceil((time_flood_new - time_flood).days/div)
                    if diff_days > interval_days*2:
                        cutted = pd.concat([cutted, dup_df, dup_df_2], axis=0)
                        continue
                    else:
                        extra_days = interval_days*2 - diff_days + 1
                        before_flood = int(extra_days/3) #give twice as much importance to data before a flood than after a flood
                        after_flood = extra_days - before_flood

                        dup_df_2 = dup_df_2.iloc[:-after_flood]
                        dup_df = dup_df.iloc[before_flood:]

                #Attach updated dataframes to cutted
                cutted = pd.concat([cutted, dup_df, dup_df_2], axis = 0, ignore_index=True) #ignore index here????

    return cutted, block