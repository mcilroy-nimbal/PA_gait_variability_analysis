import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import pingouin as pg
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
subj_details = subj_sum[subj_sum['subject_stat'] == 'mean_days']
subj_details = subj_details[['subject_id','group','sa_class']]

file2 = "SA-PR01_daily_summary.csv"
subj_daily = pd.read_csv(drive+path+file2)
#checking if means are means of mean per day or mean or steps/days?
med_by_subj = subj_daily.groupby('subject_id')[['sedentary', 'moderate','total_steps','se_total','sleep_duration_total']].median()
med_by_subj = med_by_subj.rename(columns={'moderate': 'mvpa'})
med_by_subj = med_by_subj.merge(subj_details[['subject_id', 'group', 'sa_class']], on='subject_id', how='left')
med_by_subj.loc[med_by_subj['sa_class'] == '100+ SuperAger', 'sa_class'] = 'SuperAger'
subj_med = med_by_subj.loc[med_by_subj['sa_class'].isin(['Control', 'SuperAger'])]

sd_by_subj = subj_daily.groupby('subject_id')[['sedentary', 'moderate','total_steps','se_total','sleep_duration_total']].std()
sd_by_subj = sd_by_subj.rename(columns={'moderate': 'mvpa'})
sd_by_subj = sd_by_subj.merge(subj_details[['subject_id', 'group', 'sa_class']], on='subject_id', how='left')
sd_by_subj.loc[sd_by_subj['sa_class'] == '100+ SuperAger', 'sa_class'] = 'SuperAger'
subj_sd = sd_by_subj.loc[sd_by_subj['sa_class'].isin(['Control', 'SuperAger'])]
print (len(subj_med), len(subj_sd))

save_file=False
within_day=False
all_subjs=True

#################################
#range of within day variability
#################################

if within_day:
    var_table1 = subj_med[['sedentary', 'mvpa','total_steps','se_total','sleep_duration_total']].agg(['min','max']).T
    var_table2 = subj_sd[['sedentary', 'mvpa','total_steps','se_total','sleep_duration_total']].agg(['min','max']).T

    var_list = ['sedentary', 'mvpa','total_steps','se_total','sleep_duration_total']
    df_list = [subj_med, subj_sd]
    for index, df1 in enumerate(df_list):
        if index == 0:
            label='median'
        else:
            label='std'
        print(label)

        for var in var_list:
            # Group data using groupby and extract values
            groups = df1.groupby('sa_class')[var].apply(list)
            # Perform independent t-test
            t_stat, p_value = ttest_ind(*groups, equal_var=False)  # Unpacking the groups
            # Cohen's d calculation
            cohen_d = pg.compute_effsize(groups['Control'], groups['SuperAger'], eftype='cohen')
            print(f"{var}:\t {t_stat:.3f},\t {p_value:.3f},\t {cohen_d:.3f}")
        # Controls subjects
        print("\r")
        print ('CONTROL')
        table = df1.loc[df1['sa_class'] == 'Control']
        print(table[['sedentary', 'mvpa', 'total_steps', 'se_total', 'sleep_duration_total']].agg(['count', 'median', 'std']))
        print('SUPERAGER')
        table = df1.loc[df1['sa_class'] == 'SuperAger']
        print(table[['sedentary', 'mvpa', 'total_steps', 'se_total', 'sleep_duration_total']].agg(['count', 'median', 'std']))
        print('ALL')
        print(df1[['sedentary', 'mvpa', 'total_steps', 'se_total', 'sleep_duration_total']].agg(['count', 'median', 'std']))
    print ('done')
#All subjects
if all_subjs:
    all_med_table = subj_med[['sedentary', 'mvpa','total_steps','se_total','sleep_duration_total']].agg(['count', 'median', 'std'])
    print (all_med_table)
    #wear_table = df1[['ankle_wear_duration', 'wrist_wear_duration', 'chest_wear_duration']].agg(['count','median', 'std']).T
    #activity_table = subj_sum_sub1[['sedentary', 'light', 'mvpa']].agg(['count','median', 'std']).T
    #gait_table = subj_sum_sub1[['total_steps', 'bouted_steps', 'unbouted_steps','total_walking_duration']].agg(['count','median', 'std']).T
    #sleep_table = subj_sum_sub1[['se_total','sptw_duration_total', 'sleep_duration_total']].agg(['count','median', 'std']).T
    #final_by_group_table = pd.concat([wear_table, activity_table, gait_table, sleep_table])

    #if save_file:
    #    #Write to CSV file
    #    final_by_group_table.to_csv(drive + path + "AIC2025_data_ALL_median_results2.csv")

    # Group data using groupby and extract values
    #groups = final_by_group_table.groupby('Group')['Value'].apply(list)

    # Perform independent t-test
    #t_stat, p_value = ttest_ind(*groups, equal_var=False)  # Unpacking the groups


    #Controls subjects
    #controls = df1.loc[df1['sa_class'] == 'Control']
    #con_med_table = controls[['sedentary', 'mvpa', 'total_steps', 'se_total', 'sleep_duration_total']].agg(
    #                    ['count', 'median', 'std'])
    #wear_table = controls[['ankle_wear_duration', 'wrist_wear_duration', 'chest_wear_duration']].agg(['count','median', 'std']).T
    #activity_table = controls[['sedentary', 'light', 'mvpa']].agg(['count','median', 'std']).T
    #gait_table = controls[['total_steps', 'bouted_steps', 'unbouted_steps','total_walking_duration']].agg(['count','median', 'std']).T
    #sleep_table = controls[['se_total', 'sptw_duration_total','sleep_duration_total']].agg(['count','median', 'std']).T
    #final_controls = pd.concat([wear_table, activity_table, gait_table, sleep_table])
    #if save_file:
        # Write to CSV file
        #final_controls.to_csv(drive + path + "AIC2025_data_CONTROL_median_results2.csv")

    #superagers
    #sa = df1.loc[df1['sa_class'] == 'SuperAger']
    #sa_med_table = sa[['sedentary', 'mvpa', 'total_steps', 'se_total', 'sleep_duration_total']].agg(
    #    ['count', 'median', 'std'])

    #SA = subj_sum_sub1.loc[subj_sum_sub1['sa_class'] == 'SuperAger']
    #wear_table = SA[['ankle_wear_duration', 'wrist_wear_duration', 'chest_wear_duration']].agg(['count','mean', 'std']).T
    #activity_table = SA[['sedentary', 'light', 'mvpa']].agg(['count','mean', 'std']).T
    #gait_table = SA[['total_steps', 'bouted_steps', 'unbouted_steps','total_walking_duration']].agg(['count','mean', 'std']).T
    #sleep_table = SA[['se_total','sptw_duration_total', 'sleep_duration_total']].agg(['count','mean', 'std']).T
    #final_sa = pd.concat([wear_table, activity_table, gait_table, sleep_table])
    #if save_file:
        # Write to CSV file
    #    final_sa.to_csv(drive + path + "AIC2025_data_SA_results2.csv")
     #   print ('pause')
print ('done')

