import pandas as pd
import numpy as np
from datetime import datetime, time
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glob


def create_bin_density_files(study, root, nimbal_drive, paper_path, master_subj_list, version):
    # check - but use this one - \prd\nimbalwear\OND09
    if study == 'OND09':
        path1 = root + '\\nimbalwear\\OND09\\analytics\\'
    elif study == 'SA-PRO1':
        path1 = root + '\\nimbalwear\\SA-PR01-022\\data\\'
    else:
        breakpoint()

    log_out_path = nimbal_drive + paper_path + 'Log_files\\'
    summary_path = nimbal_drive + paper_path + 'Summary_data\\'

    nw_path = 'nonwear\\daily_cropped\\'
    bout_path = 'gait\\bouts\\'
    step_path = 'gait\\steps\\'
    daily_path = 'gait\\daily\\'
    sptw_path = 'sleep\\sptw\\'

    # PART A - loop and do bin counts

    # set the bin widths fro step/strides counting
    bin_list_steps = [5, 10, 25, 50, 100, 300]
    bin_width_time = [5, 10, 30, 60, 180, 600]

    # create header
    bin = []
    for k in range(len(bin_list_steps)):
        new = f'{'<'}_{bin_list_steps[k]}'
        bin.append(new)
    last = '>_' + str(bin_list_steps[len(bin_list_steps) - 1])
    bin.append(last)
    part1 = ['n_' + item for item in bin]
    bin_list_steps_header = part1
    part2 = ['strides_' + item for item in bin]
    bin_list_steps_header.extend(part2)

    bin = []
    for k in range(len(bin_width_time)):
        new = f'{'<'}_{bin_width_time[k]}'
        bin.append(new)
    last = '>_' + str(bin_width_time[len(bin_width_time) - 1])
    bin.append(last)
    part1 = ['n_' + item for item in bin]
    bin_width_time_header = part1
    part2 = ['strides_' + item for item in bin]
    bin_width_time_header.extend(part2)

    basic = ['subj', 'visit', 'date', 'wear', 'group', 'all/sleep', 'daily_total', 'total', 'not_bouted']
    steps_header = basic + bin_list_steps_header
    width_header = basic + bin_width_time_header

    # create blank panda dataframe for summary data
    steps_summary = pd.DataFrame(columns=steps_header)
    width_summary = pd.DataFrame(columns=width_header)

    # TODO fix the walkign at night/sleep calculation

    for j, subject in enumerate(master_subj_list):
        visit = '01'
        print('Subject: ' + subject)

        # get step data for subject
        try:
            steps = pd.read_csv(path1 + step_path + subject + '_' + visit + '_GAIT_STEPS.csv')
        except:
            continue
        try:
            bouts = pd.read_csv(path1 + bout_path + subject + '_' + visit + '_GAIT_BOUTS.csv')
        except:
            bouts = None
        try:
            daily = pd.read_csv(path1 + daily_path + subject + '_' + visit + '_GAIT_DAILY.csv')
        except:
            continue
        try:
            sleep_file = path1 + sptw_path + subject + '_' + visit + '_SPTW.csv'
            sleep = pd.read_csv(sleep_file)
            found_sleep = True
        except:
            found_sleep = False
        try:
            temp = path1 + nw_path + subject + '*_NONWEAR_DAILY.csv'
            match = glob.glob(temp)
            ankle_nw = [file for file in match if 'Ankle' in file]
            file = ankle_nw[0]
            nw_data = pd.read_csv(file)
        except:
            continue

        # drop duplicate columns from daily before merge with NW
        daily.drop(['study_code', 'subject_id', 'coll_id', 'date', ], axis=1, inplace=True)

        # combine nonwear and daily steps by day_num
        merged_daily = pd.merge(nw_data, daily, on='day_num')

        # remove days that are only partial (nwear <70000?)
        # minimimum of 20 hours of wear time
        merged_daily = merged_daily[merged_daily['wear'] > 72000]  # 86400 secs in 24 hours
        merged_daily['date'] = pd.to_datetime(merged_daily['date'])
        merged_daily['date'] = merged_daily['date'].dt.date

        # reset sleep to day, wake, to bed
        if found_sleep:
            new_sleep = wake_sleep(sleep)
        else:
            new_sleep = None

        ###############################################################
        # creates bins
        steps_summary, width_summary = steps_by_day(steps_summary, steps, bin_list_steps, width_summary,
                                                    bouts, bin_width_time, merged_daily, found_sleep, new_sleep,
                                                    subject, visit, group='all')

        print('processing.....')
        ##############################################################
        # runs density function for each subject and day
        time_sec = 60
        data = step_density_sec(steps, merged_daily, time_sec)
        data.to_csv(summary_path + 'density\\' + subject + '_' + visit + '_' + version + '_'+ str(time_sec) + 'sec_density.csv')

        ##############################################################
        # runs stride time for each subejct and day
        data = stride_time_interval(steps, merged_daily)
        data.to_csv(summary_path + 'stride_time\\' + subject + '_' + version + '_' + visit + '_stride_time.csv')

    # write bins file summary
    steps_summary.to_csv(summary_path + study + '_' + version + '_bout_steps_daily_bins_with_unbouted.csv', index=False)
    width_summary.to_csv(summary_path + study + '_' + version + '_bout_width_daily_bins_with_unbouted.csv', index=False)

    print('done')

    return


def select_subjects(nimbal_drive, study):
    # which subjects to analyze
    demodata = read_demo_data(nimbal_drive, study)
    master_subj_list = demodata['SUBJECT']
    demodata['master_subj_list'] = demodata['SUBJECT'].apply(lambda x: '_'.join(x.split('_')[-2:]))

    ########################################################
    # loop through each eligible subject
    # File the time series in a paper specific forlder?
    master_subj_list = []
    for i, subject in enumerate(demodata['SUBJECT']):
        if study == 'OND09':
            parts = subject.split('_', 2)  # Split into at most 3 parts
            if len(parts) == 3:
                subject = parts[0] + '_' + parts[1] + parts[2]  # Recombine without the second underscore
        elif study == 'SA-PR01':
            subject = 'SA-PR01_' + subject
        master_subj_list.append(subject)
        #print(f'\rFind subjs - Progress: {i}' + ' of ' + str(len(demodata)), end='', flush=True)

        # find non-wear to see if enough data
        # first find the Ankle, Wrist and other nonwear
        # temp = path1 + nw_path + subject + '*_NONWEAR_DAILY.csv'
        # match = glob.glob(temp)
        # ankle_list = [file for file in match if 'Ankle' in file]
        # wrist_list = [file for file in match if 'Wrist' in file]
        # chest_list = [file for file in match if 'Chest' in file]
        # if len(ankle_list) < 1:
        #    log_file.write('file: ' + subject + ' no ankle nonwear file' +'\n')
        #    continue
        # elif len(ankle_list) > 2:
        #    #select the first ? if there are more than 1
        #    log_file.write('file: ' + subject + ' 2 ankle non-wear - took 1st' + '\n')

    # select subject list#
    # if study == 'SA-PR01':
    #    sub_study = 'AAIC 2025'
    #    subjects = pd.read_csv(summary_path+'subject_ids_'+sub_study+'.csv')
    #    master_subj_list = subjects['SUBJECT']

    return master_subj_list

def read_demo_ondri_data(path):
    ###########################################
    # read in the cleaned data file for the HANNDS methods paper
    # this woudl read in the elegible subejcts with demogrpahic data
    # demodata = read_orig_clean_demo()
    # Import data files - use this if file already created
    demodata = pd.read_csv(path + "OND09_ALL_01_CLIN_DEMOG_2025_CLEAN_HANDDS_METHODS_N245.csv")

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

    for day_time in unique_days:
        day = pd.to_datetime(day_time).date()
        wake, bed = find_time_sleep(day, sleep_data)
        print ('day: \t'+str(day)+ '\twake \t'+str(wake)+' \tbed \t'+str(bed))
        rows.append({'day': day, 'wake': wake, 'bed': bed})
    new_sleep = pd.DataFrame(rows)

    return new_sleep


def find_time_sleep(day, sleep_data):
    temp = sleep_data[sleep_data['bed_day'] == day]
    temp = temp.reset_index(drop=True)
    if len(temp) == 0:
        bed = datetime.combine(day, time(23, 59))
    elif len(temp) == 1:
        bed = temp.loc[0,'start_time']
    elif len(temp) > 1:
        #select the one that is latest but does not have and short < 1 hour duration)
        temp['sptw_dur'] = temp['end_time'] - temp['start_time']
        # Filter rows with sleep duration >= 1 hour
        filtered_df = temp[temp['sptw_dur'] >= pd.Timedelta(hours=1)]
        if len(filtered_df) == 0:
            x = temp.loc[temp['start_time'].idxmax()]
        else:
            # Find the row with the latest wake time
            x = filtered_df.loc[filtered_df['start_time'].idxmax()]
        bed = x['start_time']
    temp2 = sleep_data[sleep_data['wake_day'] == day]
    temp2 = temp2.reset_index(drop=True)
    if len(temp2) == 0:
        wake = datetime.combine(day, time(00, 00))
    elif len(temp2) == 1:
        wake = temp2.loc[0,'end_time']
    elif len(temp2) > 1:
        # select the one that is latest but does not have and short < 1 hour duration)
        temp2['sptw_dur'] = temp2['end_time'] - temp2['start_time']
        # Filter rows with sleep duration >= 1 hour
        filtered_df = temp2[temp2['sptw_dur'] >= pd.Timedelta(hours=1)]
        if len(filtered_df) == 0:
            x = temp2.loc[temp2['end_time'].idxmax()]
        else:
            # Find the row with the latest wake time
            x = filtered_df.loc[filtered_df['end_time'].idxmax()]
        wake = x['end_time']
    if bed < wake:
        bed = datetime.combine(day, time(23, 59))
    return wake, bed

def steps_by_day (steps_summary, steps, bin_list_steps, width_summary, bouts, bin_width_time,
                  merged_daily, found_sleep, new_sleep, subject, visit, group='all'):

    #loop through days all steps
    for i, row in merged_daily.iterrows():
        wear = row['wear']
        curr_day = row['date']
        daily_tot_steps = row['total_steps']

        #sleep row
        if found_sleep:
            sleep1 = new_sleep[new_sleep['day'] == curr_day]
            if not sleep1.empty:
                sleep1 = sleep1.reset_index(drop=True)
                wake = sleep1.loc[0,'wake']
                bed = sleep1.loc[0,'bed']
            else:
                found_sleep = False

        #print(' #: '+str(i)+" -"+ str(curr_day), end=" ")
        steps['date'] = pd.to_datetime(steps['step_time']).dt.date
        all_steps = steps[steps['date'] == curr_day]
        all_steps = all_steps.reset_index(drop=True)
        all_steps['step_time'] = pd.to_datetime(all_steps['step_time'])

        #count total number of steps
        total_steps = len(all_steps)
        if found_sleep:
            temp = all_steps[(all_steps['step_time'] <= wake) | (all_steps['step_time'] >= bed)]
            total_steps_sleep = len(temp)
        else:
            total_steps_sleep = -1

        #unbouted steps
        temp = all_steps[all_steps['gait_bout_num'] == 0]
        unbouted = len(temp)

        if found_sleep:
            sleep_temp = temp[(temp['step_time'] <= wake) | (temp['step_time'] >= bed)]
            unbouted_sleep = len(sleep_temp)
        else:
            unbouted_sleep = -1

        bouts['date'] = pd.to_datetime(bouts['start_time']).dt.date
        bouts['start_time'] = pd.to_datetime(bouts['start_time'])
        bouts['end_time'] = pd.to_datetime(bouts['end_time'])

        all_bouts = bouts[bouts['date'] == curr_day]

        if found_sleep:
            sleep_bouts = all_bouts[((all_bouts['start_time'] <= wake) & (all_bouts['end_time'] <= wake)) |
                                    ((all_bouts['end_time'] >= bed) & (all_bouts['end_time'] >= bed))]

        # bin by steps
        bin_count = []
        bin_sum = []
        bin_count_sleep = []
        bin_sum_sleep = []
        temp_list = bin_list_steps + [9999]
        for i, step in enumerate(temp_list):

            #select only this rows that meet criteria
            ed = step
            if i == 0:
                st = 0
            elif i == 9999:
                st = step
                ed = 9999
            else:
                st = bin_list_steps[i-1]

            #daytime
            temp = all_bouts[(all_bouts['step_count'] > st) & (all_bouts['step_count'] <= ed)]
            bin_count.append(len(temp))
            bin_sum.append(temp['step_count'].sum())

            if found_sleep:
                #night time
                temp = sleep_bouts[(sleep_bouts['step_count'] > st) & (sleep_bouts['step_count'] <= ed)]
                bin_count_sleep.append(len(temp))
                bin_sum_sleep.append(temp['step_count'].sum())
            else:
                bin_count_sleep.append(-1)
                bin_sum_sleep.append(-1)

        new_row = [subject, visit, curr_day, wear, group, 'all', daily_tot_steps, total_steps, unbouted, *bin_count, *bin_sum]
        steps_summary.loc[len(steps_summary)] = new_row

        new_row_sleep = [subject, visit, curr_day, wear, group, 'sleep', daily_tot_steps, total_steps_sleep, unbouted_sleep, *bin_count_sleep, *bin_sum_sleep]
        steps_summary.loc[len(steps_summary)] = new_row_sleep

        #bin by duration
        bin_width_count = []
        bin_width_sum = []
        bin_width_count_sleep = []
        bin_width_sum_sleep = []
        temp_list = bin_width_time + [9999]
        for i, step in enumerate(temp_list):

            #select only the rows that meet criteria
            ed = step
            if i == 0:
                st = 0
            elif i == 9999:
                st = step
                ed = 9999
            else:
                st = bin_width_time[i-1]

            #wake
            all_bouts['bout_dur'] =(all_bouts['end_time'] - all_bouts['start_time']).dt.total_seconds()
            temp = all_bouts[(all_bouts['bout_dur'] > st) & (all_bouts['bout_dur'] <= ed)]
            bin_width_count.append(len(temp))
            bin_width_sum.append(temp['step_count'].sum())

            if found_sleep:
                #night time
                sleep_bouts['bout_dur'] = (sleep_bouts['end_time'] - sleep_bouts['start_time']).dt.total_seconds()
                temp = sleep_bouts[(sleep_bouts['bout_dur'] > st) & (sleep_bouts['bout_dur'] <= ed)]
                bin_width_count_sleep.append(len(temp))
                bin_width_sum_sleep.append(temp['step_count'].sum())
            else:
                bin_width_count_sleep.append(-1)
                bin_width_sum_sleep.append(-1)

        new_row = [subject, visit, curr_day, wear, group, 'all', daily_tot_steps, total_steps,
                   unbouted, *bin_width_count, *bin_width_sum]
        width_summary.loc[len(width_summary)] = new_row

        new_row_sleep = [subject, visit, curr_day, wear, group, 'sleep', daily_tot_steps, total_steps_sleep,
                   unbouted_sleep, *bin_width_count_sleep, *bin_width_sum_sleep]
        width_summary.loc[len(width_summary)] = new_row_sleep

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

def stride_time_interval(steps, merged_daily):
    #creastes a summary of stride tiem intervals
    # loop through days
    #header_days = [f'day_{i + 1}' for i in range(len(merged_daily))]
    data = pd.DataFrame()#columns=header_days)
    count=0
    for i, row in merged_daily.iterrows():

        curr_day = row['date']
        steps['date'] = pd.to_datetime(steps['step_time']).dt.date
        all = steps[steps['date'] == curr_day]

        #caluuclaet difference in time bewteen consectuve strides
        stride_time = pd.to_datetime(all['step_time'])
        diff = stride_time.diff().dt.total_seconds()
        min_array = list(diff)

        #need to pad with NaNs if not matchign lenth of other days
        length = len(min_array)
        current_len = len(data)

        # Extend the DataFrame with NaN rows if needed
        if length > current_len:
            # Add extra rows
            new_rows = length - current_len
            data = pd.concat([data, pd.DataFrame([np.nan] * new_rows)], ignore_index=True)

        # Pad the data with NaNs if it's shorter than current DataFrame
        if length < current_len:
            min_array = min_array + [np.nan] * (current_len - length)

        # Add the column
        data[f"Day {count} - {curr_day.strftime('%Y-%m-%d')}"] = min_array
        count = count+1

    data = data.drop(index=0).drop(columns=data.columns[0])

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

def read_demo_data (drive, study):
    ###########################################
    # read in the cleaned data file for the HANNDS methods paper
    if study == 'OND09':
        new_path = drive+'\\Papers_NEW_April9\\Shared_Common_data\\OND09\\'
        demodata = read_demo_ondri_data(new_path)
    elif study == 'SA-PR01':
        new_path = drive +'\\SuperAging\\data\\summary\\AAIC_2025\\SA-PR01_collections.csv'
        demodata = pd.read_csv(new_path)
        demodata['AGE'] = demodata['age']
        demodata['SUBJECT'] = demodata['subject_id']
        demodata['COHORT'] = demodata['group']
        demodata['EMPLOYMENT STATUS'] = None

    return demodata

def corr_matrix_all_columns(data, para):

    # Select numeric columns only
    df_numeric = data.select_dtypes(include=[np.number])
    cols = df_numeric.columns

    # Create combined matrix
    combined_matrix = pd.DataFrame(index=cols, columns=cols)

    # Calculate correlation and p-value
    for col1 in cols:
        for col2 in cols:
            if para == True:
                r, p = pearsonr(df_numeric[col1], df_numeric[col2])
            else:
                r, p = spearmanr(df_numeric[col1], df_numeric[col2])
            combined_matrix.loc[col1, col2] = f"r={r:.2f}, p={p:.3f}"

    return combined_matrix

def clustering (data, ncluster):

    #Normalize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    #Cluster with KMeans
    kmeans = KMeans(n_clusters=ncluster, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    data['cluster'] = kmeans.fit_predict(scaled_data)

    #Melt the DataFrame into long-form
    data_out = data.melt(id_vars='cluster', var_name='feature', value_name='value')
    return data_out, labels

def create_table(data, cont_vars, categ_vars):
    # ---- Continuous variables: mean ± std ----
    cont_summary = (data[cont_vars].agg(['count', 'mean', 'std']).round(4))
    cont_summary = cont_summary.T.reset_index()
    cont_summary.columns = ['variable', 'count', 'mean', 'std']
    def categ_pert_table(df, categorical_cols):
        all_tables = []

        for col in categorical_cols:
            value_counts = df[col].value_counts(dropna=False)
            percentages = df[col].value_counts(normalize=True, dropna=False).mul(100).round(1)
            summary_pct = pd.DataFrame({'variable': col,'category': value_counts.index,'count': value_counts.values,
                                        'percent': percentages.values})
            all_tables.append(summary_pct)
        # Combine all summaries
        pct = pd.concat(all_tables, ignore_index=True)
        # Optional: rearrange columns
        return pct

    # ---- Categorical variables: %
    categ_summary = categ_pert_table(data, categ_vars)

    return categ_summary, cont_summary

def get_demo_characteristics(study, sub_study):
    # characteristics
    new_path = 'W:\\SuperAging\\data\\summary\\' + sub_study + '\\conference\\'
    file1 = study + '_collections.csv'
    demodata = pd.read_csv(new_path + file1)
    demodata = demodata[demodata['has_wearables_demographics'] == True]

    # list of variables to tabulate
    var_list = ['subject_id', 'group', 'sa_class', 'age_at_visit', 'sex', 'race', 'educ', 'mc_employment_status',
                'maristat', 'livsitua', 'independ', 'lsq_total', 'global_psqi', 'adlq_totalscore',
                'currently_exercise', 'currently_exercise_specify']
    demodata = demodata[var_list]

    ##################################################################
    # select certain group members
    demodata = demodata[demodata['group'].isin(['control', 'superager'])].reset_index()
    demodata.rename(columns={'group': 'GROUP'}, inplace=True)
    demodata.rename(columns={'subject_id': 'SUBJECT'}, inplace=True)

    # rename
    demodata['sex'] = demodata['sex'].replace({'1': 'Male', '2': 'Female', '3': 'Non-binary'})
    demodata['mc_employment_status'] = demodata['mc_employment_status'].replace(
        {1: 'Full-time', 2: 'Part-time', 3: 'Retired', 4: 'Disabled', 5: 'NA or never worked'})
    demodata['maristat'] = demodata['maristat'].replace(
        {'1': 'Married', '2': 'Widowed', '3': 'Divorced', '4': 'Separated',
         '5': 'Never married', '6': 'Living as married/domestic partner', '9': 'Unknown'})
    demodata['livsitua'] = demodata['livsitua'].replace(
        {'1': 'Lives alone', '2': 'Lives (1) spouse or partner', '3': 'Lives (1) other',
         '4': 'Lives with caregiver not spouse/partner, relative, or friend',
         '5': 'Lives with a group private residence',
         '6': 'Lives in group home', '9': 'Unknown'})
    demodata['independ'] = demodata['independ'].replace(
        {1: 'Able to live independently', 2: 'Assist-complex activities',
         3: 'Assist-basic activities',
         4: 'Assist-complete', 9: 'Unknown'})
    demodata['currently_exercise_specify'] = demodata['currently_exercise_specify'].replace(
        {1: 'Every day', 2: 'At least 3x / week', 3: '1 x week', 4: '< once a week', 5: '< 1 a month'})
    demodata['currently_exercise'] = demodata['currently_exercise'].replace(
        {1: 'Yes', 2: 'No'})
    demodata['race'] = demodata['race'].replace(
        {1: 'White', 2: 'Black', 3: 'American Indian or Alaska Native', 4: 'Native Hawaiian',
         5: 'Asian', 50: 'Other', 99: 'Unknown'})
    #demodata['residenc'] = demodata['residenc'].replace(
    #    {1: 'Single - or multi-family private residence', 2: 'Retirement community or independent group living',
    #     3: 'Assisted living, adult family home', 4: 'Skilled nursing facility', 9: 'Unknown'})
    #demodata['demosresidenceruralurban'] = demodata['demosresidenceruralurban'].replace(
    #    {1: 'Rural', 2: 'Urban'})


    return demodata