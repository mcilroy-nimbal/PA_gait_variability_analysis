#THIS CODE: CREATES BIN FILES AND DENSITY DATA FROM ELIGIBLE SUBJECTS
#Output Files locate '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'Summary_data\\''
#this runns all data available in analytics fiels located int he target directory

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from Functions import wake_sleep, steps_by_day, step_density_sec,read_demo_ondri_data, read_demo_data, stride_time_interval
import numpy as np
import seaborn as sns
import datetime
import openpyxl
import warnings
warnings.filterwarnings("ignore")

#study = 'OND09'
study = 'SA-PR01'

#set up paths
root = 'W:'

#check - but use this one - \prd\nimbalwear\OND09
if study == 'OND09':
    path1 = root+'\\NiMBaLWEAR\\OND09\\analytics\\'
else:
    path1 = root + '\\nimbalwear\\SA-PR01-022\\data\\'
nimbal_drive = 'O:'
paper_path = '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'
log_out_path = nimbal_drive + paper_path + 'Log_files\\'
summary_path = nimbal_drive + paper_path + 'Summary_data\\'

nw_path = 'prep\\nonwear\\daily_cropped\\'

bout_path = 'analytics\\gait\\bouts\\'
step_path = 'analytics\\gait\\steps\\'
daily_path = 'analytics\\gait\\daily\\'
sptw_path = 'analytics\\sleep\\sptw\\'

# files to log details of processing
curr_date = datetime.datetime.now().strftime('%Y_%m_%d')
filen = f'{'log_read_step_bouts_'}_{curr_date}.txt'
log_file = open(log_out_path + filen, 'w')
y = 0

#demo_path = nimbal_drive+'\\Papers_NEW_April9\\Shared_Common_data\\'+study+'\\'
#demodata = read_demo_data(demo_path)


########################################################
# loop through each eligible subject
# File the time series in a paper specific forlder?
#master_subj_list = []
#for i, subject in enumerate(demodata['SUBJECT']):
#    # just foir SA
#    if study =='OND09':
#        parts = subject.split('_', 2)  # Split into at most 3 parts
#        if len(parts) == 3:
#            subject = parts[0] + '_' + parts[1] + parts[2]  # Recombine without the second underscore
#    elif study =='SA-PR01':
#        subject='SA-PR01_'+subject

#    print(f'\rFind subjs - Progress: {i}' + ' of ' + str(len(demodata)), end='', flush=True)

    #find non-wear to see if enough data
    # first find the Ankle, Wrist and other nonwear
#    temp = path1 + nw_path + subject + '*_NONWEAR_DAILY.csv'
#    match = glob.glob(temp)

#    ankle_list = [file for file in match if 'Ankle' in file]
    #wrist_list = [file for file in match if 'Wrist' in file]
    #chest_list = [file for file in match if 'Chest' in file]

#    if len(ankle_list) < 1:
#        log_file.write('file: ' + subject + ' no ankle nonwear file' +'\n')
#        continue
#    elif len(ankle_list) > 2:
        #select the first ? if there are more than 1
#        log_file.write('file: ' + subject + ' 2 ankle non-wear - took 1st' + '\n')

#    master_subj_list.append(subject)

#slect subject list
sub_study = 'AAIC 2025'
subjects = pd.read_csv(summary_path+'subject_ids_'+sub_study+'.csv')
master_subj_list = subjects['SUBJECT']

log_file.write('Total # subjects: '+str(len(master_subj_list)) + '\n\n')

#PART A - loop and do bin counts

#set the bin widths fro step/strides counting
bin_list_steps = [5, 10, 25, 50, 100, 300]
bin_width_time = [5, 10, 30, 60, 180, 600]

#create header
bin=[]
for k in range(len(bin_list_steps)):
    new = f'{'<'}_{bin_list_steps[k]}'
    bin.append(new)
last = '>_' + str(bin_list_steps[len(bin_list_steps)-1])
bin.append(last)
part1 = ['n_' + item for item in bin]
bin_list_steps_header = part1
part2 = ['strides_' + item for item in bin]
bin_list_steps_header.extend(part2)

bin=[]
for k in range(len(bin_width_time)):
    new = f'{'<'}_{bin_width_time[k]}'
    bin.append(new)
last = '>_' + str(bin_width_time[len(bin_width_time)-1])
bin.append(last)
part1 = ['n_' + item for item in bin]
bin_width_time_header = part1
part2 = ['strides_' + item for item in bin]
bin_width_time_header.extend(part2)

basic = ['subj','visit','date','wear', 'group', 'all/sleep', 'daily_total', 'total', 'not_bouted']
steps_header = basic + bin_list_steps_header
width_header = basic + bin_width_time_header

#create blank panda dataframe for summary data
steps_summary = pd.DataFrame(columns=steps_header)
width_summary = pd.DataFrame(columns=width_header)

#TODO fix the walkign at night/sleep calculation

log_file.write('Part A - step counts in bins \n')

for j, subject in enumerate(master_subj_list):
    visit = '01'
    print('Subject: ' + subject)
    log_file.write('Subject: ' + subject + '\n')

    #get step data for subject
    try:
        steps = pd.read_csv(path1 + step_path + study+'_' + subject + '_' + visit + '_GAIT_STEPS.csv')
    except:
        log_file.write('\tGAIT_STEPS file not found \n')
        continue
    try:
        bouts = pd.read_csv(path1 + bout_path + study+'_' + subject + '_' + visit + '_GAIT_BOUTS.csv')
    except:
        log_file.write('\tGAIT_BOUTS not found\n')
        bouts = None
    try:
        daily = pd.read_csv(path1 + daily_path + study+'_'+subject + '_'+ visit + '_GAIT_DAILY.csv')
    except:
        log_file.write('\tGAIT_DAILY file not found \n')
        continue

    try:
        sleep_file = path1 + sptw_path + study+'_'+subject + '_' + visit + '_SPTW.csv'
        sleep = pd.read_csv(sleep_file)
        found_sleep = True
    except:
        log_file.write('\tSPTW_file not found \n')
        found_sleep = False
    try:
        temp = path1 + nw_path + study+'_'+subject + '*_NONWEAR_DAILY.csv'
        match = glob.glob(temp)
        ankle_nw = [file for file in match if 'Ankle' in file]
        file = ankle_nw[0]
        nw_data = pd.read_csv(file)
    except:
        log_file.write('\tNONWEAR DAILY file not found \n')
        continue

    # drop duplicate columns from daily before merge with NW
    daily.drop(['study_code', 'subject_id', 'coll_id', 'date', ], axis=1, inplace=True)

    # combine nonwear and daily steps by day_num
    merged_daily = pd.merge(nw_data, daily, on='day_num')
    log_file.write('\tNumber of days' + str(len(merged_daily)) + '\n')

    # remove days that are only partial (nwear <70000?)
    # minimimum of 20 hours of wear time
    merged_daily = merged_daily[merged_daily['wear_duration'] > 72000]  # 86400 secs in 24 hours
    merged_daily['date'] = pd.to_datetime(merged_daily['date'])
    merged_daily['date'] = merged_daily['date'].dt.date
    log_file.write('\tNumber of days with > 72000 secs wear'+str(len(merged_daily))+'\n')

    # reset sleep to day, wake, to bed
    if found_sleep:
        new_sleep = wake_sleep(sleep)
    else:
        new_sleep = None

    ###############################################################
    #creates bins
    steps_summary, width_summary = steps_by_day(log_file, steps_summary, steps, bin_list_steps, width_summary, bouts, bin_width_time, merged_daily, found_sleep, new_sleep, subject, visit, group='all')

    print ('processing.....')

    ##############################################################
    #runs density function for each subejct and day
    time_sec=60
    data = step_density_sec(steps, merged_daily, time_sec)
    data.to_csv(summary_path+'density\\'+subject+'_'+visit+'_'+str(time_sec)+'sec_density.csv')

    ##############################################################
    #runs stride time for each subejct and day
    data = stride_time_interval(steps, merged_daily)
    data.to_csv(summary_path+'stride_time\\'+subject+'_'+visit+'_stride_time.csv')

# write bins file summary
steps_summary.to_csv(summary_path + study+'_bout_steps_daily_bins_with_unbouted.csv', index=False)
width_summary.to_csv(summary_path + study+'_bout_width_daily_bins_with_unbouted.csv', index=False)

log_file.close()
print('done')


##########################
'''



    ###########################################################
    # PART A - bins - wake and sleep
    # find nonwear and sleep time data
    # only want data for 24 days midnight to midnight for pattern data
    # so need the start and stop from first midnight to last midnight of data
    ###########################################################
    sleep['end_time'] = pd.to_datetime(sleep['end_time'])
    sleep['start_time'] = pd.to_datetime(sleep['start_time'])
    sleep['relative_date'] = pd.to_datetime(sleep['relative_date'])
    steps['step_time'] = pd.to_datetime(steps['step_time'])
    steps['date'] = steps['step_time'].dt.date


    # re-aligning sleep so that wake and sleep on same calendar day on same line
    # needed for the 24 hour midnight to midnight approach with steps
    # first occurence of date (relative_date)
    #       - 1st part is midnight to end_time on day before
    #       - 2nd is start_time to midnight
    # if there is more than one find last one  with correct relative_date
    #       - take the
    #       - 1st part is midnight to start
    #use fundtion to find wake and sleep for each day for this subjects tabualr sptw data
    new_sleep = wake_sleep(sleep)

    y = y + 0.05
    #loop through days
    for i, row in merged_daily.iterrows():
        curr_day = row['date']
        print(' #: '+str(i)+" -"+ str(curr_day), end=" ")
        #curr_day = pd.to_datetime(row['date'])
        #select only rows that match the date (from daily)
        all = steps[steps['date'] == curr_day]

        #count total number of steps
        total_steps = len(all)

        #finds steps in SPTW
        sleep_row = new_sleep[new_sleep['day'] == curr_day]
        if len(sleep_row) < 1:
            log_file.write('    Error - no sleep data on day - '+str(i)+ ' ' + str(curr_date)+'\n')
            continue
        else:
            #keep all that are earlie than new-sleep wake
            #keep all that are later than new_sleep[end_time']
            first = all['step_time'] < sleep_row.iloc[0].loc['wake']  #True for before wake up
            last = all['step_time'] > sleep_row.iloc[0].loc['bed']  #True if after go to bed

            #use fits and all to select on sleep steps from all
            sleep_steps = all[(first | last)]  # take all TRUE
            n_sleep_steps = first.sum() + last.sum() # count all TRUE

            #sort by bouts
            wake = ~(first | last) #Take all FALSE
            n_wake_steps = wake.sum() #count all FALSE
            wake_steps = all[wake] #take all FALSE for wake steps

            # log details
            log_file.write('  '+str(curr_date)+ '  Wake: '+ str(sleep_row.iloc[0].loc['wake']) + '   Bed: '+ str(sleep_row.iloc[0].loc['bed'])
                           + ' N sleep steps: '+ str(n_sleep_steps) + '   Wake steps: '+ str(n_wake_steps)+ '\n')

            new_row = [subject, visit, curr_day, total_steps, n_sleep_steps, n_wake_steps, sleep_row['wake'], sleep_row['bed']]
            new_row_series = pd.DataFrame([new_row], columns=summary.columns)
            summary = pd.concat([summary, new_row_series], ignore_index=True)

            # steps within each bout bin
            # create bout_bin (steps within bouts - from the step file) - set the bout windws
            #bow windows passed as list and also names the bout_bins header
            #sleep_bouts = sleep_bouts.append(bout_bin)
            bout_bin = bout_bins(sleep_steps, bin_list)
            sleep_bouts.loc[len(sleep_bouts)] = [subject, visit, curr_day, *bout_bin]
            #non_sleep bouts
            bout_bin = bout_bins(wake_steps, bin_list)
            non_sleep_bouts.loc[len(non_sleep_bouts)] = [subject, visit, curr_day, *bout_bin]

_timelog_file.close()

stats = pd.DataFrame(columns=['subj','ndays', 'total_med', 'total_std', 'total_max',
                              'sleep_med', 'sleep_std', 'sleep_max',
                              'wake_med', 'wake_std', 'wake_max'])
print(len(summary))
subj = summary['subj'].unique()
n_subjs = len(subj)
for i in subj:
    temp = summary[summary['subj']==i]
    ndays = len(temp)
    total_steps_med = temp['total_steps'].median()
    total_steps_std = temp['total_steps'].std()
    total_steps_max = temp['total_steps'].max()
    sleep_steps_med = temp['sleep_steps'].median()
    sleep_steps_std = temp['sleep_steps'].std()
    sleep_steps_max = temp['sleep_steps'].max()
    wake_steps_med = temp['wake_steps'].median()
    wake_steps_std = temp['wake_steps'].std()
    wake_steps_max = temp['wake_steps'].max()
    new_row = [i, ndays, total_steps_med, total_steps_std, total_steps_max,
               sleep_steps_med, sleep_steps_std, sleep_steps_max,
               wake_steps_med, wake_steps_std, wake_steps_max]
    new_row_series = pd.DataFrame([new_row], columns=stats.columns)
    stats = pd.concat([stats, new_row_series], ignore_index=True)

#write stats file to CSV
out_file = 'stats1_feb25.csv'
stats.to_csv(summary_path+out_file, index=False)
out_file2 = 'wake_bouts_feb25.csv'
non_sleep_bouts.to_csv(summary_path+out_file2, index=False)
out_file3 = 'sleep_bouts_feb25.csv'
sleep_bouts.to_csv(summary_path+out_file3, index=False)
print ('done')



            if raw_plot:
                ###############################
                #plots the step times for all steps
                #plotting each row adds 1 to Y?
                #X is time (exldue date)
                y = y+0.01
                #non_sleep_steps['time'] = non_sleep_steps['step_time'].dt.time
                hours = ((wake_steps['step_time'].dt.hour) + (wake_steps['step_time'].dt.minute / 60)
                         + (wake_steps['step_time'].dt.second / 3600))# + (non_sleep_steps['step_time'].dt.microsecond) / 3600000)
                yarray= [y] * len(hours)
                #First non-sleep
                plt.scatter(hours, yarray, color='orange',s=1)
                #sleep
                hours = ((sleep_steps['step_time'].dt.hour) + (sleep_steps['step_time'].dt.minute / 60)
                         + (sleep_steps['step_time'].dt.second / 3600))  # + (sleep_steps['step_time'].dt.microsecond) / 3600000)
                yarray = [y] * len(hours)
                plt.scatter(hours, yarray, color='blue', s=1)
plt.show()


'''