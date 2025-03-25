import pandas as pd
import os
import matplotlib.pyplot as plt
from Functions import wake_sleep, bout_bins
import numpy as np
import seaborn as sns
import datetime
import openpyxl


root = 'W:'
path1 = root+'\\NiMBaLWEAR\\OND09\\analytics\\'
nimbal_drive = 'O:'
out_path = nimbal_drive +'\\Student_Projects\\gait_pattern_paper_feb2024\\'

''' 
read gait files - steps and bouts 
read sleep files  - for sleep time window - classify night time stepping
read nonwear for wear time and days to use - only 24 days??
Loop for all peopel with data available 
- criteria - need sleep, and gait for X days?
'''

'''
read in the cleaned data file for the HANNDS methods paper for the number of
'''

#Import data files
demodata = pd.read_csv("W:/OND09 (HANDDS-ONT)/HANDDS methods paper data/Data/LabKey Data/OND09_RELEASE_CLIN files/OND09_ALL_01_CLIN_DEMOG/OND09_ALL_01_CLIN_DEMOG_2023JUL04_DATA.csv")
scrndata = pd.read_csv("W:/OND09 (HANDDS-ONT)/HANDDS methods paper data/Data/LabKey Data/OND09_RELEASE_CLIN files/OND09_ALL_01_CLIN_SCRN/OND09_ALL_01_CLIN_SCRN_2023SEP13_DATA.csv")
pptlist = pd.read_excel("W:/OND09 (HANDDS-ONT)/HANDDS methods paper data/Data/Number of Days_By Sensor_Bill outputs_20Aug2024_WithCohorts.xlsx")







#Review non-wear for subjects that match criteria
nw_path = 'nonwear\\daily_cropped\\'
bout_path = 'gait\\bouts\\'
step_path = 'gait\\steps\\'
daily_path = 'gait\\daily\\'
sptw_path = 'sleep\\sptw\\'

file_list = os.listdir(path1+nw_path)
#select only files - no directories
file_list = [file for file in file_list if os.path.isfile(os.path.join(path1+nw_path, file))]
print ('N of all files: ' + str(len(file_list)))
file_list = [file for file in file_list if 'Ankle' in file]
print ('N of Ankle files: ' + str(len(file_list)))

#Find Ankle files with enough data - midnight to midnight
step1 = pd.DataFrame()

#create blank panda dataframe for summary data
summary = pd.DataFrame(columns=['subj','visit', 'date', 'total_steps', 'sleep_steps', 'wake_steps','wake','bed'])
#set the bin widths fro step/strides counting
bin_list = [3, 5, 10, 20, 50, 100, 300, 600]

# convert bin list to header
str_bin_list=[]
for i in range(len(bin_list)):
  new = f'{'<'}_{bin_list[i]}'
  str_bin_list.append(new)
last = '>_' + str(bin_list[len(bin_list)-1])
str_bin_list.append(last)
header = ['subj','visit','date']
header.extend(str_bin_list)

#create empty data frames
non_sleep_bouts = pd.DataFrame(columns=header)
sleep_bouts = pd.DataFrame(columns=header)

# files to log details of processing
curr_date = datetime.datetime.now().strftime('%Y_%m_%d')
filen = f'{'log_read_step_bouts_'}_{curr_date}.txt'
log_file = open (out_path + filen, 'w')
y = 0
#loop through files
for file in file_list:

    nw_data = pd.read_csv(path1+nw_path+file)
    print ('\nFile: '+file, end=" ")
    # only data with 7 days
    if len(nw_data) < 7:
        log_file.write('file: '+ file + ' less than 7 days in NWEAR file  - ndays = '+str(len(nw_data)) +'\n')
        continue
    log_file.write('File: ' + file + '  OK - NON WEAR'+'\n')

    file_noext = file.split(".")[0]
    parts = file.split("_")
    subj = parts[1]
    visit = parts[2]
    file_start = parts[0]+"_"+ parts[1]+"_"+parts[2]
    daily = pd.read_csv(path1 + daily_path + file_start+ '_GAIT_DAILY.csv')
    steps = pd.read_csv(path1 + step_path + file_start + '_GAIT_STEPS.csv')
    sleep = pd.read_csv(path1 + sptw_path + file_start + '_SPTW.csv')

    #ignore if sleep is not avaibale (OR we shoudl just do totals??)
    if len(sleep) < (len(daily) -2):
        log_file.write('    Error - not enough sleep data:   len-sleep -' + str(len(sleep))+ ' len-daily - '+ str(len(daily))+'\n')
        continue
    sleep['end_time'] = pd.to_datetime(sleep['end_time'])
    sleep['start_time'] = pd.to_datetime(sleep['start_time'])
    sleep['relative_date'] = pd.to_datetime(sleep['relative_date'])
    steps['step_time'] = pd.to_datetime(steps['step_time'])
    steps['date'] = steps['step_time'].dt.date

    # drop duplicate columns from daily before merge with NW
    daily.drop(['study_code','subject_id','coll_id','date',], axis=1, inplace=True)

    #combine nonwear and daily steps by day_num
    merged_daily = pd.merge(nw_data, daily, on='day_num')

    #remove days that are only partial (nwear <70000?)
    merged_daily = merged_daily[merged_daily['wear'] > 79200]  #86400 secs in 24 hours
    merged_daily['date'] = pd.to_datetime(merged_daily['date'])
    merged_daily['date'] = merged_daily['date'].dt.date

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

            new_row = [subj, visit, curr_day, total_steps, n_sleep_steps, n_wake_steps, sleep_row['wake'], sleep_row['bed']]
            new_row_series = pd.DataFrame([new_row], columns=summary.columns)
            summary = pd.concat([summary, new_row_series], ignore_index=True)

            # steps within each bout bin
            # create bout_bin (steps within bouts - from the step file) - set the bout windws
            #bow windows passed as list and also names the bout_bins header
            #sleep_bouts = sleep_bouts.append(bout_bin)
            bout_bin = bout_bins(sleep_steps, bin_list)
            sleep_bouts.loc[len(sleep_bouts)] = [subj, visit, curr_day, *bout_bin]
            #non_sleep bouts
            bout_bin = bout_bins(wake_steps, bin_list)
            non_sleep_bouts.loc[len(non_sleep_bouts)] = [subj, visit, curr_day, *bout_bin]

'''
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

#write the different files to data location

log_file.close()

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
stats.to_csv(out_path+out_file, index=False)
out_file2 = 'wake_bouts_feb25.csv'
non_sleep_bouts.to_csv(out_path+out_file2, index=False)
out_file3 = 'sleep_bouts_feb25.csv'
sleep_bouts.to_csv(out_path+out_file3, index=False)
print ('done')
