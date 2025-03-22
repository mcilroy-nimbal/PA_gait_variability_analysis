import pandas as pd
import matplotlib.pyplot as plt


'''
######################################
# AAIC2025
#
#variables: sedentary time (mean + SD/)  min/day
            mod/vig time (mean + SD/)  min/day 

######################################
'''
drive = 'w:\\'
path =  'SuperAging\\data\\summary\\AAIC 2025\\'
file1 = "SA-PR01_subject_summary.csv"
subj_sum = pd.read_csv(drive+path+file1)

#Select only Control and SuperAger for 'sa_class' column
#MAKE SURE TO SELECT 100+ AS WELL
subj_sum.loc[subj_sum['sa_class'] == '100+ SuperAger', 'sa_class'] = 'SuperAger'
subj_sum_sub = subj_sum.loc[subj_sum['sa_class'].isin(['Control', 'SuperAger'])]

#Select only mean_days from 'subj_stat' colummn
subj_sum_sub1 = subj_sum_sub[subj_sum_sub['subject_stat'] == 'mean_days']
'''
#All subjects
wear_table = subj_sum_sub1[['ankle_wear_duration', 'wrist_wear_duration', 'chest_wear_duration']].agg(['count','mean', 'std']).T
activity_table = subj_sum_sub1[['sedentary', 'light', 'mvpa']].agg(['count','mean', 'std']).T
gait_table = subj_sum_sub1[['total_steps', 'bouted_steps', 'unbouted_steps','total_walking_duration']].agg(['count','mean', 'std']).T
sleep_table = subj_sum_sub1[['se_total','sptw_duration_total', 'sleep_duration_total']].agg(['count','mean', 'std']).T
final__by_group_table = pd.concat([wear_table, activity_table, gait_table, sleep_table])
# Write to CSV file
final__by_group_table.to_csv(drive + path + "AIC2025_data_ALL_results.csv")
'''

#Controls subjects
controls = subj_sum_sub1.loc[subj_sum_sub1['sa_class'] == 'Control']
wear_table = controls[['ankle_wear_duration', 'wrist_wear_duration', 'chest_wear_duration']].agg(['count','mean', 'std']).T
activity_table = controls[['sedentary', 'light', 'mvpa']].agg(['count','mean', 'std']).T
gait_table = controls[['total_steps', 'bouted_steps', 'unbouted_steps','total_walking_duration']].agg(['count','mean', 'std']).T
sleep_table = controls[['se_total', 'sptw_duration_total','sleep_duration_total']].agg(['count','mean', 'std']).T
final_controls = pd.concat([wear_table, activity_table, gait_table, sleep_table])
# Write to CSV file
final_controls.to_csv(drive + path + "AIC2025_data_CONTROL_results.csv")

#superagers
SA = subj_sum_sub1.loc[subj_sum_sub1['sa_class'] == 'SuperAger']
wear_table = SA[['ankle_wear_duration', 'wrist_wear_duration', 'chest_wear_duration']].agg(['count','mean', 'std']).T
activity_table = SA[['sedentary', 'light', 'mvpa']].agg(['count','mean', 'std']).T
gait_table = SA[['total_steps', 'bouted_steps', 'unbouted_steps','total_walking_duration']].agg(['count','mean', 'std']).T
sleep_table = SA[['se_total','sptw_duration_total', 'sleep_duration_total']].agg(['count','mean', 'std']).T
final_sa = pd.concat([wear_table, activity_table, gait_table, sleep_table])
# Write to CSV file
final_sa.to_csv(drive + path + "AIC2025_data_SA_results.csv")




