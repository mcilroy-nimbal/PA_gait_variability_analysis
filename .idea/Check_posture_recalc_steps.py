import pandas as pd
import os
import matplotlib.pyplot as plt

root = 'W:'
path1 = root+'\\NiMBaLWEAR\\OND09\\analytics\\'
nimbal_drive = 'O:'
out_path = nimbal_drive +'\\Student_Projects\\gait_pattern_paper_feb2024\\'

path_sensors = root+'\\NiMBaLWEAR\\OND09\\wearables\\device_edf_cropped\\'

#Review non-wear for subjects that match criteria
step_path = 'gait\\steps\\'

#search for fiels with GAIT_STEPS in filename
file_list = os.listdir(path1+step_path)
#select only files - no directories
file_list = [file for file in file_list if os.path.isfile(os.path.join(path1+step_path, file))]
print ('N of all files: ' + str(len(file_list)))
file_list = [file for file in file_list if 'GAIT_STEPS' in file]
print ('N of Step files: ' + str(len(file_list)))


#loop trhough file list
for file in file_list:

    file_noext = file.split(".")[0]
    parts = file.split("_")
    subj = parts[1]
    visit = parts[2]
    file_start = parts[0] + "_" + parts[1] + "_" + parts[2]

    #daily = pd.read_csv(path1 + daily_path + file_start + '_GAIT_DAILY.csv')
    #sleep = pd.read_csv(path1 + sptw_path + file_start + '_SPTW.csv')

    ## read steps files
    steps = pd.read_csv(path1 + step_path + file_start + '_GAIT_STEPS.csv')
    step_data = pd.read_csv(path1+step_path+file)
    # Ensure timestamp column is a datetime type and set as index
    step_data['timestamp'] = pd.to_datetime(step_data['step_time'])
    step_data.set_index('timestamp', inplace=True)
    print ('\nFile: '+file, end=" ")
    start_time = step_data.iloc[0]['step_time']
    end_time = step_data.iloc[-1]['step_time']
    print (start_time, end_time)

    #read sensor file
    #bitium ? for prone?
    #read ankle for horizontal
    rawfile = study + "_" + subj + "_" + visit + '_' + sensor + '_' + loc + ".edf"
    print('reading sensor file # :'+ rawfile)
    # read raw data header
    device.import_edf(file_path=path_sensor + rawfile)
    #accel_x_sig = device.get_signal_index('Accelerometer x')
    #accel_y_sig = device.get_signal_index('Accelerometer y')
    accel_z_sig = device.get_signal_index('Accelerometer z')
    cropped = device.crop()
    #x = cropped.signals[accel_x_sig]
    #y = cropped.signals[accel_y_sig]
    z = cropped.signals[accel_z_sig]
    #t = cropped.get_timestamps(accel_x_sig)
    sample_rate.append(device.signal_headers[accel_z_sig]['sample_rate'])
    start_date.append(device.header['start_datetime'])
    max_len.append(len(z))






#Find Ankle files with enough data - midnight to midnight
step1 = pd.DataFrame()





#####################################################################


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
