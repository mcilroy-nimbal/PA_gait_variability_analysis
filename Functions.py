import pandas as pd
import numpy as np
from datetime import datetime, time


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

    #Interleave start and end time rows
    #use this to find wake and bed for each day
    sleep_data = sleep_data[sleep_data['overnight'] == True]
    sleep_data['start_time'] = pd.to_datetime(sleep_data['start_time'])
    sleep_data['end_time'] = pd.to_datetime(sleep_data['end_time'])
    sleep_data['bed_day'] = sleep_data['start_time'].dt.date
    sleep_data['wake_day'] = sleep_data['end_time'].dt.date

    rows=[]
    unique_days = sleep_data['relative_date'].unique()
    #unique_days = unique_days[:-1]
    for day in unique_days:
        day1 = pd.to_datetime(day).date()
        temp = sleep_data[sleep_data['bed_day'] == day1]
        temp = temp.reset_index(drop=True)

        if len(temp) == 0:
            bed = datetime.combine(day1, time(23, 59))
        elif len(temp) == 1:
            bed = temp.loc[0,'start_time']
        else:
            last = temp.loc[temp['start_time'].idxmax()]
            bed = last['start_time']

        temp1 = sleep_data[sleep_data['wake_day'] == day1]
        temp1 = temp1.reset_index(drop=True)
        if len(temp1) == 0:
            wake = None
        elif len(temp1) == 1:
            wake = temp1.loc[0, 'end_time']
        else:
            target = pd.to_timedelta('09:00:00')
            temp1['time_diff'] = temp1['end_time'].dt.time.apply(lambda t: abs(pd.to_timedelta(str(t)) - target))
            close = temp1.loc[temp1['time_diff'].idxmin()]
            wake = close['end_time']

        rows.append({'day': day1, 'wake': wake, 'bed': bed})
    new_sleep = pd.DataFrame(rows)

    return new_sleep

def steps_by_day (steps_summary, steps, bin_list_steps, width_summary, bouts, bin_width_time,
                  merged_daily, new_sleep, subject, visit, group='all'):

    #loop through days all steps
    for i, row in merged_daily.iterrows():
        wear = row['wear']
        curr_day = row['date']
        daily_tot_steps = row['total_steps']

        #sleep row
        sleep1 = new_sleep[new_sleep['day'] == curr_day]
        sleep1 = sleep1.reset_index(drop=True)
        wake = sleep1.loc[0,'wake']
        bed = sleep1.loc[0,'bed']

        #print(' #: '+str(i)+" -"+ str(curr_day), end=" ")
        steps['date'] = pd.to_datetime(steps['step_time']).dt.date
        all_steps = steps[steps['date'] == curr_day]
        all_steps = all_steps.reset_index(drop=True)
        all_steps['step_time'] = pd.to_datetime(all_steps['step_time'])

        #count total number of steps
        total_steps = len(all_steps)
        temp = all_steps[(all_steps['step_time'] <= wake) | (all_steps['step_time'] >= bed)]
        total_steps_sleep = len(temp)

        #unbouted steps
        temp = all_steps[all_steps['gait_bout_num'] == i]
        #TODO: fix this - unbouted = len(temp)
        sleep_temp = temp[(temp['step_time'] <= wake) | (temp['step_time'] >= bed)]
        #TODo: and this unbouted_sleep = len(sleep_temp)

        bouts['date'] = pd.to_datetime(bouts['start_time']).dt.date
        bouts['start_time'] = pd.to_datetime(bouts['start_time'])
        bouts['end_time'] = pd.to_datetime(bouts['end_time'])

        all_bouts = bouts[bouts['date'] == curr_day]
        sleep_bouts = all_bouts[((all_bouts['start_time'] <= wake) & (all_bouts['end_time'] <= wake)) |
                                ((all_bouts['end_time'] >= bed) & (all_bouts['end_time'] >= bed))]

        # bin by steps
        bin_count=[]
        bin_count_sleep=[]
        bin_list_steps.append(9999)
        for i, step in enumerate(bin_list_steps):

            #select only this rows that meet criteria
            ed = step
            if i == 0:
                st = 0
            else:
                st=bin_list_steps[i-1]
            temp = all_bouts[(all_bouts['step_count'] > st) & (all_bouts['step_count'] <= ed)]
            bin_count.append(len(temp))

            #night time
            temp = sleep_bouts[(sleep_bouts['step_count'] > st) & (sleep_bouts['step_count'] <= ed)]
            bin_count_sleep.append(len(temp))

        new_row = [subject, visit, curr_day, wear, group, 'all', daily_tot_steps, total_steps, unbouted, *bin_count]
        steps_summary.loc[len(steps_summary)] = new_row

        new_row_sleep = [subject, visit, curr_day, wear, group, 'sleep', daily_tot_steps, total_steps_sleep, unbouted_sleep, *bin_count_sleep]
        steps_summary.loc[len(steps_summary)] = new_row_sleep

        bin_width_count = []
        bin_width_count_sleep = []
        bin_width_time.append(99999)
        for i, step in enumerate(bin_width_time):

            #select only the rows that meet criteria
            ed = step
            if i == 0:
                st = 0
            else:
                st=bin_list_steps[i-1]
            all_bouts['bout_dur'] =(all_bouts['end_time'] - all_bouts['start_time']).dt.total_seconds()
            temp = all_bouts[(all_bouts['bout_dur'] > st) & (all_bouts['bout_dur'] <= ed)]
            bin_width_count.append(len(temp))

            #night time
            sleep_bouts['bout_dur'] = (sleep_bouts['end_time'] - sleep_bouts['start_time']).dt.total_seconds()
            temp = sleep_bouts[(sleep_bouts['bout_dur'] > st) & (sleep_bouts['bout_dur'] <= ed)]
            bin_width_count_sleep.append(len(temp))

        new_row = [subject, visit, curr_day, wear, group, 'all', daily_tot_steps, total_steps, unbouted, *bin_width_count]
        width_summary.loc[len(steps_summary)] = new_row

        new_row_sleep = [subject, visit, curr_day, wear, group, 'sleep', daily_tot_steps, total_steps_sleep, unbouted_sleep, *bin_width_count_sleep]
        width_summary.loc[len(steps_summary)] = new_row_sleep

        # steps within each bout bin
        # create bout_bin (steps within bouts - from the step file) - set the bout windws
        #by steps - windows passed as list and also names the bout_bins header
        #bout_bin_steps, not_bouted = bout_bins_steps(all_steps, bin_list)
        #steps_summary.loc[len(all_steps)] = [subject, visit, curr_day, wear, group,total_steps, not_bouted, *bout_bin_steps]


    return steps_summary, width_summary

'''def bout_bins_steps (data, bin_list):
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
'''

def step_density_sec(steps, merged_daily, time_sec):

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
        full_range= int(86400 / time_sec)
        for step in range(full_range):
            start = step * time_sec
            end = (step+1) * time_sec
            sub = all[(all['time_sec'] > start) & (all['time_sec'] <= end)]
            min_array.append(len(sub))
        data[header_days[count]] = [curr_day] + min_array

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

def summary_density_bins(data):
    # set frequency of bins that meet cut points
    zero_tot = np.sum(data == 0)
    vlow_tot = np.sum((data > 0) & (data <= 5))
    low_tot = np.sum((data > 5) & (data <= 20))
    med_tot = np.sum((data > 20) & (data <= 40))
    high_tot = np.sum(data > 40)

    long_thresh = 40
    long = data > long_thresh
    padded = np.pad(long.astype(int), (1, 1), mode='constant')
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    stops = np.where(diff == -1)[0]
    total = []
    bout_3 = 0
    bout_3_dur = 3
    bout_10 = 0
    bout_10_dur = 10
    for s, e in zip(starts, stops):
        if e-s >= bout_3_dur:
            bout_3 += 1
        if e-s >= bout_10_dur:
            bout_10 += 1

    out_tot = [len(data), zero_tot, vlow_tot, low_tot, med_tot, high_tot, bout_3, bout_10]
    return out_tot

def read_demo_data (study):
    ###########################################
    # read in the cleaned data file for the HANNDS methods paper
    if study == 'OND09':
        nimbal_dr = 'o:'
        new_path = '\\Papers_NEW_April9\\Shared_Common_data\\OND09\\'
        demodata = read_demo_ondri_data(nimbal_dr, new_path)
    elif study == 'SA-PR01':
        new_path = 'W:\\SuperAging\\data\\summary\\RPPR 2025\\SA-PR01_collections.csv'
        demodata = pd.read_csv(new_path)
        demodata['AGE'] = demodata['age']
        demodata['SUBJECT'] = demodata['subject_id']
        demodata['COHORT'] = demodata['group']
        demodata['EMPLOYMENT STATUS'] = None
    return demodata

