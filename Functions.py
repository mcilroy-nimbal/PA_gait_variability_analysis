import pandas as pd
import numpy as np

def wake_sleep (sleep_data):
    #rules - wake
    # - earliest endTime on curr day that does have a sleep time start < 1 hr

    new_sleep = pd.DataFrame(columns=['day', 'wake', 'bed'])

    #Interleave start and end time rows
    #use this to find wake and bed for each day
    sleep_data['bed_day'] = sleep_data['start_time'].dt.date
    sleep_data['wake_day'] = sleep_data['end_time'].dt.date
    sleep_data['bed_time'] = sleep_data['start_time'].shift(-1) # bring the next row start to row with the wake
    sleep_data['wake_time'] = sleep_data['end_time']

    no_bed = sleep_data.iloc[len(sleep_data)-1].loc['bed_time'] == 'NaT'
    no_wake = sleep_data.iloc[len(sleep_data)-1].loc['wake_time'] == 'NaT'
    if not no_bed or not no_wake:
        #drop the row
        sleep_data = sleep_data.iloc[:-1]

    unique_days = sleep_data['wake_day'].unique()

    new_sleep['day'] = unique_days

    for index, day in enumerate(unique_days):

        new_sleep.loc[index, 'day'] = day
        temp = sleep_data[sleep_data['wake_day'] == day]
        temp.reset_index(drop=True, inplace=True)

        b_index = 0 #bount row index
        #if only 1 wake on day accept
        if len(temp) == 1:
            new_sleep.loc[index,'wake'] = temp.iloc[b_index].loc['wake_time']
            new_sleep.loc[index, 'bed'] = temp.iloc[b_index].loc['bed_time']

        # if more than 1 wake select one the meets criteria
        else:
            #find row with wake
            wake=0
            for j in range(len(temp)):
                time2 = temp.iloc[j].loc['bed_time'].time()
                bed_hour = (time2.hour * 3600 + time2.minute * 60 + time2.second) / 3600

                time1 = temp.iloc[j].loc['wake_time'].time()
                wake_hour = (time1.hour * 3600 + time1.minute * 60 + time1.second) / 3600

                diff = bed_hour - wake_hour

                if wake == 0 and wake_hour < 4.0:
                    continue
                elif wake_hour < 12.0 and diff > 1:
                    wake = j
                else:
                    continue

            if wake == -1:
                print ('error - no wake found'), #this is OK if last row - so corrected later by removign last row
                continue
            else:
                new_sleep.loc[index, 'wake'] = temp.iloc[wake].loc['wake_time']


            #why not just grab the last and check to see if the previous event was within an hour?
            temp_backwards = temp.iloc[::-1]
            temp_backwards.reset_index(drop=True, inplace=True)

            bed = -1
            for j in range(len(temp)):

                time2 = temp_backwards.iloc[j].loc['bed_time'].time()
                bed_hour = (time2.hour * 3600 + time2.minute * 60 + time2.second) / 3600
                #bedtime is p[ast midnight add 24 to bed_hour
                bed_day = temp_backwards.iloc[j].loc['bed_time'].date()
                if bed_day != day:
                    bed_hour = bed_hour+24

                time1 = temp_backwards.iloc[j].loc['wake_time'].time()
                wake_hour = (time1.hour * 3600 + time1.minute * 60 + time1.second) / 3600
                diff = bed_hour - wake_hour

                if diff > 2:
                    bed = j
                else:
                    continue

            if bed == -1:
                print ('error - no bed found'),
            else:
                new_sleep.loc[index, 'bed'] = temp_backwards.iloc[bed].loc['bed_time']

        #print(new_sleep.loc[index,])

    #new_sleep = new_sleep.drop(new_sleep.index[-1]) #remove the last day sicne no bed-time

    return new_sleep

def bout_bins (data, bin_list):
    # steps within each bout bin
    # create bout_bin (steps within bouts - from the step file) - set the bout windows
    # bow windows passed as list and also names the bout_bins header
    unique_bouts = data['gait_bout_num'].unique()
    bout_bin = [0] * (len(bin_list)+1)
    for i in unique_bouts:
        temp = data[data['gait_bout_num'] == i]
        bout_len = len(temp)
        #loop through bin_list
        for j in range(len(bin_list)):
            #set start (st) and end (end)
            if j == 0:
                st = -1
            else:
                st = bin_list[j-1]
            end = bin_list[j]
            #figure out where the bout_length fits into the bin_list
            if bout_len > st and bout_len <= end:
                bout_bin[j] += bout_len
        #chaeck the last bin outside the loop - is it greater than that last bin_list value
        if bout_len > bin_list[j]:
            bout_bin[j+1] += bout_len
    return bout_bin

def step_density(steps, time, window, overlapping):


    return