import pandas as pd
import numpy as np
from datetime import datetime


def read_demo_ondri_data(drive, path):
    ###########################################
    # read in the cleaned data file for the HANNDS methods paper
    # this woudl read in the elegible subejcts with demogrpahic data
    # demodata = read_orig_clean_demo()
    # Import data files - use this if file already created
    demodata = pd.read_csv(drive + path + "OND09_ALL_01_CLIN_DEMOG_2025_CLEAN_HANDDS_METHODS_N245.csv")

    # merge dual diagonis - other MCI
    demodata['COHORT'] = demodata['COHORT'].replace('MCI;CVD', 'CVD')
    demodata['COHORT'] = demodata['COHORT'].replace('MCI;PD', 'PD')
    demodata['COHORT'] = demodata['COHORT'].replace('AD;MCI', 'MCI')
    # collapse AD MCI
    demodata['COHORT'] = demodata['COHORT'].replace('AD', 'MCI')
    demodata['COHORT'] = demodata['COHORT'].replace('MCI', 'AD/MCI')



    return demodata

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

def steps_by_day (summary, steps, merged_daily, subject, visit, bin_list, group):

    #loop through days
    for i, row in merged_daily.iterrows():
        wear = row['wear']
        curr_day = row['date']
        #print(' #: '+str(i)+" -"+ str(curr_day), end=" ")
        steps['date'] = pd.to_datetime(steps['step_time']).dt.date
        all = steps[steps['date'] == curr_day]

        #count total number of steps
        total_steps = len(all)

        # steps within each bout bin
        # create bout_bin (steps within bouts - from the step file) - set the bout windws
        #bow windows passed as list and also names the bout_bins header
        bout_bin, not_bouted = bout_bins(all, bin_list)
        summary.loc[len(all)] = [subject, visit, curr_day, wear, group,total_steps, not_bouted, *bout_bin]

    return summary

def bout_bins (data, bin_list):
    # steps within each bout bin
    # create bout_bin (steps within bouts - from the step file) - set the bout windows
    # bow windows passed as list and also names the bout_bins header
    unique_bouts = data['gait_bout_num'].unique()
    bout_bin = [0] * (len(bin_list)+1)
    for i in unique_bouts:
        temp = data[data['gait_bout_num'] == i]
        bout_len = len(temp)
        if i == 0:
            nbouted = bout_len
        else:
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
            #check the last bin outside the loop - is it greater than that last bin_list value
            if bout_len > bin_list[j]:
                bout_bin[j+1] += bout_len

    return bout_bin, nbouted

def step_density_1min(steps, merged_daily):

    # loop through days
    header_days = [f'day_{i + 1}' for i in range(len(merged_daily))]
    data = pd.DataFrame(columns=header_days)
    count=0
    for i, row in merged_daily.iterrows():

        curr_day = row['date']
        steps['date'] = pd.to_datetime(steps['step_time']).dt.date
        all = steps[steps['date'] == curr_day]
        #loop through every minute and find step count - index is minutes of day

        time = pd.to_datetime(all['step_time']).dt.time
        all['time_sec'] = time.apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
        min_array = []
        for step in range(1440):
            start = step * 60
            end = (step+1) * 60
            sub = all[(all['time_sec'] > start) & (all['time_sec'] <= end)]
            min_array.append(len(sub))
        data[header_days[count]] = min_array
        #print ('Total - '+ header_days[count] + ':  '+str(sum(min_array)))
        count = count+1
    return data

def read_orig_fix_clean_demo():
    nimbal_dr = 'o:'
    new_path = '\\Papers_NEW_April9\\Shared_Common_data\\OND09\\'

    #Import data files
    demodata_1 = pd.read_csv("W:/OND09 (HANDDS-ONT)/HANDDS methods paper data/Data/LabKey Data/OND09_RELEASE_CLIN files/OND09_ALL_01_CLIN_DEMOG/OND09_ALL_01_CLIN_DEMOG_2023JUL04_DATA.csv")
    scrndata = pd.read_csv("W:/OND09 (HANDDS-ONT)/HANDDS methods paper data/Data/LabKey Data/OND09_RELEASE_CLIN files/OND09_ALL_01_CLIN_SCRN/OND09_ALL_01_CLIN_SCRN_2023SEP13_DATA.csv")
    pptlist = pd.read_excel("W:/OND09 (HANDDS-ONT)/HANDDS methods paper data/Data/Number of Days_By Sensor_Bill outputs_20Aug2024_WithCohorts.xlsx")

    demodata = demodata_1[demodata_1['SUBJECT'].isin(pptlist['Subj'])]


    #Adjust for cohort discrepancy
    demodata.loc[demodata['SUBJECT'] == 'OND09_SBH_0060', 'cohort'] = 'MCI;CVD'
    demodata.loc[demodata['SUBJECT'] == 'OND09_SBH_0175', 'cohort'] = 'AD'
    demodata.loc[demodata['SUBJECT'] == 'OND09_SBH_0186', 'cohort'] = 'Community Dwelling'
    demodata.loc[demodata['SUBJECT'] == 'OND09_SBH_0338', 'cohort'] = 'PD'
    demodata.loc[demodata['SUBJECT'] == 'OND09_SBH_0361', 'cohort'] = 'MCI;CVD'

    #Assign participants with 2 diagnoses to 1 cohort

    def cohortrecode(cohort):
        if cohort == "AD;MCI":
            return "AD/MCI"
        elif cohort == "AD":
            return "AD/MCI"
        elif cohort == "MCI":
            return "AD/MCI"
        elif cohort == "MCI;CVD":
            return "CVD"
        elif cohort == "MCI;PD":
            return "PD"
        else:
            return cohort

    #New cohort count
    demodata["cohort"] = demodata.apply(lambda x: cohortrecode(x["cohort"]), axis=1)
    newcohortcount = demodata["cohort"].count()
    print(newcohortcount)

    newcohortcountgrouped = demodata.groupby("cohort").size()
    print(newcohortcountgrouped)

    # List of subjects to check
    subjects = ['OND09_SBH_0060', 'OND09_SBH_0175', 'OND09_SBH_0186', 'OND09_SBH_0338', 'OND09_SBH_0361']

    # Print 'newcohort' for the specified subjects
    for subject in subjects:
        print(f"Subject {subject}: cohort = {demodata.loc[demodata['SUBJECT'] == subject, 'cohort'].values[0]}")

    demodata.to_csv(nimbal_dr+new_path+'OND09_ALL_01_CLIN_DEMOG_2025_CLEAN_HANDDS_METHODS_N245.csv', index=False)
    return demodata