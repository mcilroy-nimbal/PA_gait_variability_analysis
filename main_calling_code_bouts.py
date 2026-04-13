import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from Functions import (wake_sleep, steps_by_day, step_density_sec, clustering,
                       read_demo_ondri_data, read_demo_data, stride_time_interval,
                       create_bin_files, create_density_files, select_subjects,
                       all_bouts_histogram, create_table, alpha_gini_bouts)
from Gait_bout_basic_anal_graphs import (calc_basic_stride_bouts_stats, bouts_SML)

import numpy as np
import seaborn as sns
import datetime
import openpyxl
import warnings
warnings.filterwarnings("ignore")

#set up file paths
study = 'OND09'
root = 'W:'
nimbal_drive ='O:'
paper_path = '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'
demo_path = '\\Papers_NEW_April9\\Shared_Common_data\\OND09\\'

bin_list_steps = [5, 10, 25, 50, 100, 300]
bin_width_time = [5, 10, 30, 60, 180, 600]


#values for density analysis
window_size = 15
step_size = 1

#min and max # days to include
min_days = 7
max_days = 7

#which subjects?
master_subj_list = select_subjects(nimbal_drive, study)
#master_subj_list = ['OND09_SBH0006']
print('\nTotal # subjects: \t' + str(len(master_subj_list)) + '\n')
print('First 5 subject in list...' + str(master_subj_list[:5])+'\n')

#select specific subjects from the study group
#select ONDRI
group_name = ['Community Dwelling', 'PD', 'ADMCI', 'CVD']

study = 'OND09'
path = nimbal_drive + demo_path
demodata = read_demo_ondri_data(path)

'''
#demogarphci details by group
#AGE,SEX,MRTL_STATUS,EMPLOY_STATUS, LIVING_CIRCUM
data = demodata[demodata['COHORT'].isin(group_name)]
categ, cont = create_table(data, ['AGE'], ['SEX','COHORT', 'MRTL_STATUS','EMPLOY_STATUS', 'LIVING_CIRCUM'])
for i in range(5):
    if i == 0:
        label = 'Community Dwelling'
    elif i == 1:
        label = 'PD'
    elif i == 2:
        label = 'AD/MCI'
    elif i == 3:
        label = 'CVD'
    elif i == 4:
        label = 'ALL'
    data = demodata[demodata['COHORT'] == label]
    categ, cont = create_table(data, ['AGE'], ['SEX','MRTL_STATUS','EMPLOY_STATUS', 'LIVING_CIRCUM'])
    print (group_name[i], categ, cont)
'''

#subject lists
group_lists = [0,1,2,3,4]
group_lists[0] = demodata[demodata['COHORT'] == 'Community Dwelling']['SUBJECT']
print ('Number subjects CONTROL: '+str(len(group_lists[0])))
group_lists[1] = demodata[demodata['COHORT'] == 'PD']['SUBJECT']
print('Number subjects in PD: '+str(len(group_lists[1])))
group_lists[2] = demodata[demodata['COHORT'] == 'AD/MCI']['SUBJECT']
print('Number subjects in ADMCI: '+str(len(group_lists[2])))
group_lists[3] = demodata[demodata['COHORT'] == 'CVD']['SUBJECT']
print('Number subjects in CVD: '+str(len(group_lists[3])))
group_lists[4] = demodata[demodata['COHORT'].isin(group_name)]['SUBJECT']
print('Number subjects in ALL: '+str(len(group_lists[4])))



#STEP 1 - subject # list to include
subjects = group_lists[4]
print('\nTotal # subjects in selected list: \t' + str(len(subjects)) + '\n')
create_bins = False
#Bins must be created before graphing - this creates the daily totals
#does it for all subjects on the list provided regardless of number of days
#however on days that have 20 hours will be counted (using 24hr data)
if create_bins:
    for window in ['1010', '24hr', 'wake']:
        create_bin_files(window, study, root, nimbal_drive, paper_path, subjects, bin_list_steps, bin_width_time)

#STEP 2 - SUMMARY DATA ACROSS WINDOWS AND GROUPS
#must limit to time window
#values for analysis of bins
central = 'mean'
central1 = 'Mean'
time_window =['1010','24hr','wake']
#which group to run 0-control, 1-PD, 2-AD/MCI, 3-CVD, 4 = ALL
groups = ['Control', 'PD', 'ADMCI', 'CVD', 'ALL']

calc_basic_stats = False
if calc_basic_stats:
    for index, group in enumerate(groups):
        subjects = group_lists[index]
        for window in time_window:
            step_vs_dur = False
            #calculate medians and std for each bout and clustered bouts
            calc_basic_stride_bouts_stats(step_vs_dur, nimbal_drive, study, window, paper_path, subjects, group, min_days, max_days)

            step_vs_dur = True #run duraiton file
            calc_basic_stride_bouts_stats(step_vs_dur, nimbal_drive, study, window, paper_path, subjects, group, min_days, max_days)

           # SML_median, SML_std, SML_pct_median, SML_pct_std = bouts_SML(nimbal_drive, study, window, paper_path, group)


#STEP 3 - plots and tables
subject_tables = False

create_density = False
create_stride_time = False
calc_basic_stats = False
tables1 = False
tables2 = False
figure1 = True #swarm totals
figure1b = False
figure2 = False #KDE distibiton - bouts/unbouted
figure3 = False #bout disitbution - steps and %
figure4 = False #steps per class (s,m,l)
figure4b = False #% steps per class (s,m,l)
figure5 = False # 3 coffecint of varaition - bewteen day
figure6 = False
figure7 = False #gini
common_path = nimbal_drive + paper_path + 'Created_data\\bout_bins\\'


##############################################################
# Results part 1 - 24 vs 1010 vs wake  - Figure 1A
group = 'ALL'
subj_24hr = pd.read_csv(common_path + 'summary_subject_level\\'+study + '_24hr_' + group + '_bout_duration_subj_stats.csv', header=[0, 1], skiprows=[2])
subj_pct_24hr = pd.read_csv(common_path + 'summary_subject_level\\'+study + '_24hr_' + group + '_bout_duration_pct_subj_stats.csv', header=[0, 1], skiprows=[2])
subj_1010 = pd.read_csv(common_path + 'summary_subject_level\\'+ study + '_1010_' + group + '_bout_duration_subj_stats.csv', header=[0, 1] )
subj_pct_1010 = pd.read_csv(common_path + 'summary_subject_level\\' + study + '_1010_' + group + '_bout_duration_pct_subj_stats.csv',header=[0, 1])
subj_wake = pd.read_csv(common_path + 'summary_subject_level\\' + study + '_wake_' + group + '_bout_duration_subj_stats.csv', header=[0, 1])
subj_pct_wake = pd.read_csv(common_path + 'summary_subject_level\\' + study + '_wake_' + group + '_bout_duration_pct_subj_stats.csv', header=[0, 1])

plot_24hr_all = subj_24hr[('window_total_strides', central)]
#plot_24hr_all = plot_24hr_all[1:]
plot_1010_all = subj_1010[('window_total_strides', central)]
plot_wake_all = subj_wake[('window_total_strides', central)]

#plot 1 all subjects
df = pd.DataFrame({'24 HR': plot_24hr_all, 'Wake': plot_wake_all, '10AM-10PM': plot_1010_all})
# Melt the DataFrame to long format for seaborn
melted_df = df.melt(var_name='Window', value_name='Total steps')
# Create the swarm plot
plt.figure(figsize=(6, 5))
sns.boxplot(data=melted_df, x="Window", y="Total steps", showcaps=True, hue="Window", boxprops={'facecolor': 'None'},  # transparent box so swarm is visible
        showfliers=False) # hide outliers (swarm will show them)
# Overlay swarm plot
sns.swarmplot(x='Window', y='Total steps', data=melted_df, hue="Window", size=4)
plt.ylim(bottom=0, top=12000)
plt.xlabel('Analysis Window', fontsize=14)
plt.ylabel('Average unilateral steps / day', fontsize=14)
plt.tight_layout()
plt.savefig(nimbal_drive+paper_path+"Figures_tables\\Figures\\Figure_ALL_24hr_1010_wake.png")
plt.close()

summary_tables=[]
for index, group in enumerate(groups):
    print (group)
    subjects = group_lists[index]
    temp_24hr = pd.read_csv(common_path + 'summary_subject_level\\'+study + '_24hr_' + group + '_bout_duration_subj_stats.csv', header=[0, 1], skiprows=[2])
    temp_pct_24hr = pd.read_csv(common_path + 'summary_subject_level\\'+study + '_24hr_' + group + '_bout_duration_pct_subj_stats.csv', header=[0, 1], skiprows=[2])
    temp_1010 = pd.read_csv(common_path + 'summary_subject_level\\'+ study + '_1010_' + group + '_bout_duration_subj_stats.csv', header=[0, 1] )
    temp_pct_1010 = pd.read_csv(common_path + 'summary_subject_level\\' + study + '_1010_' + group + '_bout_duration_pct_subj_stats.csv',header=[0, 1])
    temp_wake = pd.read_csv(common_path + 'summary_subject_level\\' + study + '_wake_' + group + '_bout_duration_subj_stats.csv', header=[0, 1])
    temp_pct_wake = pd.read_csv(common_path + 'summary_subject_level\\' + study + '_wake_' + group + '_bout_duration_pct_subj_stats.csv', header=[0, 1])

    temp_24hr_all = temp_24hr[('window_total_strides', central)]
    temp_24hr_all = temp_24hr_all[1:]
    temp_1010_all = temp_1010[('window_total_strides', central)]
    temp_wake_all = temp_wake[('window_total_strides', central)]
    # Example data (replace with your actual data)
    data = {"24hr": temp_24hr_all, "1010": temp_1010_all, "wake": temp_wake_all}
    # Create DataFrame
    df = pd.DataFrame(data)
    # Calculate statistics
    summary = df.agg(['count','mean', 'std', 'max', 'min'])
    summary = summary.stack().rename(group)  # rows = (stat, variable)
    summary_tables.append(summary)

final_summary = pd.concat(summary_tables, axis=1)
final_summary.to_csv(nimbal_drive+paper_path+"Figures_tables\\Tables\\Table_groups_24hr_1010_wake.csv")


################################################################################
#Fig 1 B and tables

subj_24hr = pd.read_csv(common_path + 'summary_subject_level\\'+study + '_24hr_' + group + '_bout_duration_subj_stats.csv', header=[0, 1], skiprows=[2])
subj_pct_24hr = pd.read_csv(common_path + 'summary_subject_level\\'+study + '_24hr_' + group + '_bout_duration_pct_subj_stats.csv', header=[0, 1], skiprows=[2])
plot_24hr_all = subj_24hr[('window_total_strides', central)]
plot_24hr_unbouted = subj_24hr[('window_not_bouted_strides', central)]
plot_24hr_unbouted = plot_24hr_unbouted[1:]
plot_24hr_bouted = plot_24hr_all - plot_24hr_unbouted

#plot 1 all subjects
df = pd.DataFrame({'Total': plot_24hr_all, 'Bouted': plot_24hr_bouted, 'Unbouted': plot_24hr_unbouted})
# Melt the DataFrame to long format for seaborn
melted_df = df.melt(var_name='Window', value_name='Total steps')
# Create the swarm plot
plt.figure(figsize=(6, 5))
sns.boxplot(data=melted_df, x="Window", y="Total steps", showcaps=True, hue="Window", boxprops={'facecolor': 'None'},  # transparent box so swarm is visible
        showfliers=False) # hide outliers (swarm will show them)
# Overlay swarm plot
sns.swarmplot(x='Window', y='Total steps', data=melted_df, hue="Window", size=4)
plt.ylim(bottom=0, top=12000)
plt.xlabel('Analysis Window', fontsize=14)
plt.ylabel('Average unilateral steps / day', fontsize=14)
plt.tight_layout()
plt.savefig(nimbal_drive+paper_path+"Figures_tables\\Figures\\Figure_ALL_bouted_unbouted.png")
plt.close()


######################################################################################
#Bout histogram - Total and percent steps - MEANS AND STD
plot_labels = ['Unbouted', '<5', '5-10', '10-30', '30-60', '60-180', '180-600', '>600']
summary1 = []
summary2 = []
print ('Running BIN bout analysis - means - all bouts')

for index, group in enumerate(groups):

    group_24hr = pd.read_csv(common_path + 'summary_group_level\\'+study+ '_24hr_' + group + '_bout_duration__group_stats_'+central+'.csv')
    group_pct_24hr = pd.read_csv(common_path + 'summary_group_level\\'+study+ '_24hr_' + group + '_bout_duration__pct_group_stats_'+central+'.csv')

    #plot_labels = ['Total', 'Unbouted', '<5', '5-10', '10-25', '25-50', '50-100', '100-300', '>300']
    plot_labels = ['Unbouted', '<5', '5-10', '10-30', '30-60', '60-180', '180-600', '>600']
    central_24hr_nototal = group_24hr.iloc[1:].reset_index(drop=True)
    central_24hr_pct_nototal = group_pct_24hr.iloc[1:].reset_index(drop=True)

    fig, axs = plt.subplots(2, figsize=(6, 9))
    # median std strides
    ticks = list(range(len(plot_labels)))
    axs[0].bar(central_24hr_nototal.index, central_24hr_nototal[central1].values, yerr=central_24hr_nototal['Std'], capsize=5, color='lightblue', edgecolor='black')
    #axs[0].set_title(central1 + ' unilateral steps / day by bout length', fontsize=14)
    axs[0].set_xlabel('Bout length (sec)', fontsize=14)
    axs[0].set_ylabel('Unilateral steps / day', fontsize=14)
    axs[0].set_xticks(ticks=ticks, labels=plot_labels, fontsize=10)
    axs[0].set_ylim(bottom=0, top=2250)

    # median std strides
    ticks = list(range(len(plot_labels)))
    axs[1].bar(central_24hr_pct_nototal.index, central_24hr_pct_nototal[central1].values, yerr=central_24hr_pct_nototal['Std'], capsize=5, color='lightgreen', edgecolor='black')
    #axs[1].set_title('Percent ' + central + ' unilateral steps/day by bout length', fontsize=14)
    axs[1].set_xlabel('Bout length (sec)', fontsize=14)
    axs[1].set_ylabel('% of total unilateral steps / day', fontsize=14)
    axs[1].set_xticks(ticks=ticks, labels=plot_labels, fontsize=10)
    axs[1].set_ylim(bottom=0, top=35)

    #axs.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.savefig(nimbal_drive + paper_path + "Figures_tables\\Figures\\Figure_"+group+"_bout_bins_histogram.png")
    plt.close()

    # Calculate statistics
    df_group = central_24hr_nototal.copy()
    df_group['Group'] = group
    summary1.append(df_group)
    df_group = central_24hr_pct_nototal.copy()
    df_group['Group'] = group
    summary2.append(df_group)
summary1 = pd.concat(summary1, ignore_index=True)
summary1.to_csv(nimbal_drive+paper_path+"Figures_tables\\Tables\\Table_groups_bin_bouts_total_steps.csv")
summary2 = pd.concat(summary2, ignore_index=True)
summary2.to_csv(nimbal_drive+paper_path+"Figures_tables\\Tables\\Table_groups_bin_bouts_percent_steps.csv")

#######################################################################
#Bout histogram - Intraday variation CV AND STD
summary1 = []
summary2 = []
print ('Running bout bin analysis - CVs - all bouts')

for index, group in enumerate(groups):
    #group_24hr = pd.read_csv(common_path + 'summary_group_level\\'+study+ '_24hr_' + group + '_bout_duration__group_stats_'+central+'.csv')
    #group_pct_24hr = pd.read_csv(common_path + 'summary_group_level\\'+study+ '_24hr_' + group + '_bout_duration__pct_group_stats_'+central+'.csv')
    group_24hr = pd.read_csv(common_path + 'summary_group_level\\' + study + '_24hr_' + group + '_bout_duration__group_stats_cv_' + central + '.csv')
    group_pct_24hr = pd.read_csv(common_path + 'summary_group_level\\' + study + '_24hr_' + group + '_bout_duration__pct_group_stats_cv_' + central + '.csv')

    #plot_labels = ['Total', 'Unbouted', '<5', '5-10', '10-25', '25-50', '50-100', '100-300', '>300']
    plot_labels = ['Unbouted', '<5', '5-10', '10-30', '30-60', '60-180', '180-600', '>600']
    central_24hr_nototal = group_24hr.iloc[1:].reset_index(drop=True)
    central_24hr_pct_nototal = group_pct_24hr.iloc[1:].reset_index(drop=True)

    fig, axs = plt.subplots(2, figsize=(6, 9))
    # median std strides
    ticks = list(range(len(plot_labels)))
    axs[0].bar(central_24hr_nototal.index, central_24hr_nototal[central1].values, yerr=central_24hr_nototal['Std'], capsize=5, color='lightblue', edgecolor='black')
    #axs[0].set_title(central1 + ' unilateral steps / day by bout length', fontsize=14)
    axs[0].set_xlabel('Bout length (sec)', fontsize=14)
    axs[0].set_ylabel('Intraday variability (CV) total steps / day', fontsize=14)
    axs[0].set_xticks(ticks=ticks, labels=plot_labels, fontsize=10)
    axs[0].set_ylim(bottom=0, top=2.5)

    # median std strides
    ticks = list(range(len(plot_labels)))
    axs[1].bar(central_24hr_pct_nototal.index, central_24hr_pct_nototal[central1].values, yerr=central_24hr_pct_nototal['Std'], capsize=5, color='lightgreen', edgecolor='black')
    #axs[1].set_title('Percent ' + central + ' unilateral steps/day by bout length', fontsize=14)
    axs[1].set_xlabel('Bout length (sec)', fontsize=14)
    axs[1].set_ylabel('Intraday variability (CV) % steps / day', fontsize=14)
    axs[1].set_xticks(ticks=ticks, labels=plot_labels, fontsize=10)
    axs[1].set_ylim(bottom=0, top=2.5)

    #axs.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.savefig(nimbal_drive + paper_path + "Figures_tables\\Figures\\Figure_" + group + "_CV_bout_bins_histogram.png")
    plt.close()

    # Calculate statistics
    df_group = central_24hr_nototal.copy()
    df_group['Group'] = group
    summary1.append(df_group)
    df_group = central_24hr_pct_nototal.copy()
    df_group['Group'] = group
    summary2.append(df_group)
summary1 = pd.concat(summary1, ignore_index=True)
summary1.to_csv(nimbal_drive+paper_path+"Figures_tables\\Tables\\Table_groups_bin_bouts_CV_total_steps.csv")
summary2 = pd.concat(summary2, ignore_index=True)
summary2.to_csv(nimbal_drive+paper_path+"Figures_tables\\Tables\\Table_groups_bin_bouts_CV_percent_steps.csv")



########################################################################################
#SML
print ('Running SML bout analysis - all bouts')
summary1 = []
summary2 = []

for index, group in enumerate(groups):

    subj_24hr = pd.read_csv(common_path + 'summary_subject_level\\'+study + '_24hr_' + group + '_bout_duration_subj_stats.csv', header=[0, 1], skiprows=[2])
    subj_pct_24hr = pd.read_csv(common_path + 'summary_subject_level\\'+study + '_24hr_' + group + '_bout_duration_pct_subj_stats.csv', header=[0, 1], skiprows=[2])
    plot_24hr_all = subj_24hr[('window_total_strides', central)]

    short_24hr_bouted = subj_24hr[[('strides_<_5', central), ('strides_<_10', central),('strides_<_30', central)]]
    med_24hr_bouted = subj_24hr[[('strides_<_60', central), ('strides_<_180', central)]]
    long_24hr_bouted = subj_24hr[[('strides_<_600', central), ('strides_>_600', central)]]

    plot_24hr_unbouted = subj_24hr[('window_not_bouted_strides', central)]
    plot_24hr_bouted = plot_24hr_all - plot_24hr_unbouted

    short = short_24hr_bouted.sum(axis=1)
    med = med_24hr_bouted.sum(axis=1)
    long = long_24hr_bouted.sum(axis=1)
    unbouted = plot_24hr_unbouted

    short_pct = 100*(short / plot_24hr_all)
    med_pct = 100 * (med / plot_24hr_all)
    long_pct = 100 * (long / plot_24hr_all)
    unbouted_pct = 100 * (plot_24hr_unbouted / plot_24hr_all)

    df_pct = pd.DataFrame({'Unbouted': unbouted_pct, 'Short': short_pct, 'Medium': med_pct, "Long": long_pct})
    melted_df_pct = df_pct.melt(var_name='Bout class', value_name='% Total steps')

    df_tot = pd.DataFrame({'Unbouted': unbouted, 'Short': short, 'Medium': med, "Long": long})
    melted_df_tot = df_tot.melt(var_name='Bout class', value_name='% Total steps')

    plt.figure(figsize=(8, 6))
    # Create the swarm plot
    sns.boxplot(data=melted_df_pct, x="Bout class", y="% Total steps", showcaps=True, hue="Bout class",
                boxprops={'facecolor': 'None'},  # transparent box so swarm is visible
                showfliers=False)  # hide outliers (swarm will show them)
    # Overlay swarm plot
    sns.swarmplot(x='Bout class', y='% Total steps', data=melted_df_pct, hue="Bout class",
                  palette=["red", "magenta", "blue", "green"], size=4)

    plt.ylim(bottom=0)
    #plt.title('Group: '+grp)
    plt.xlabel('Bout classification', fontsize=14)

    plt.ylabel('% of total average unilateral steps / day', fontsize=14)
    plt.tight_layout()
    plt.savefig(nimbal_drive + paper_path + "Figures_tables\\Figures\\Figure_" + group + "_SML_percent_swarm_plot.png")
    plt.close()




    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)

    sns.regplot(x=plot_24hr_all, y=plot_24hr_unbouted, color='red', label='Unbouted',
                scatter_kws={"s": 20, "alpha": 0.7}, line_kws={"color": "grey"}, ci=None, ax=axs[0, 0])
    sns.regplot(x=plot_24hr_all, y=short, color='magenta', label='Short <30 sec',
                scatter_kws={"s": 20, "alpha": 0.7}, line_kws={"color": "grey"}, ci=None, ax=axs[0, 1])
    sns.regplot(x=plot_24hr_all, y=med, color='blue', label='Medium 30-180 sec',
                scatter_kws={"s": 20, "alpha": 0.7}, line_kws={"color": "grey"}, ci=None, ax=axs[1, 0])
    sns.regplot(x=plot_24hr_all, y=long, color='green', label='Long > 180 sec',
                scatter_kws={"s": 20, "alpha": 0.7}, line_kws={"color": "grey"}, ci=None, ax=axs[1, 1])

    plt.suptitle('Total steps versus bouted and unbouted steps')

    axs[0, 0].set_ylabel(central + ' daily steps')
    axs[1, 0].set_ylabel(central + ' daily steps')
    axs[0, 0].set_xlabel('')
    axs[0, 1].set_xlabel('')
    axs[1, 0].set_xlabel('Total steps (' + central + ') / day')
    axs[1, 1].set_xlabel('Total steps (' + central + ') / day')
    axs[0, 0].set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(nimbal_drive + paper_path + "Figures_tables\\Figures\\Figure_" + group + "_SML_scatter_plots.png")
    plt.close()

    data1 = {"Group:": group, "Unbouted": unbouted, "Short": short, "Medium": med, "Long": long}
    summary = pd.DataFrame({
        "mean": [np.nanmean(pd.to_numeric(v, errors="coerce")) for v in data1.values()],
        "median": [np.nanmedian(pd.to_numeric(v, errors="coerce")) for v in data1.values()],
        "std": [np.nanstd(pd.to_numeric(v, errors="coerce")) for v in data1.values()],
        "n": [np.count_nonzero(~np.isnan(pd.to_numeric(v, errors="coerce"))) for v in data1.values()]}, index=data1.keys())
    summary1.append(summary)

    data2 = {"Group": group, "Unbouted": unbouted_pct, "Short": short_pct, "Medium": med_pct, "Long": long_pct}
    summary = pd.DataFrame({"mean": [np.nanmean(pd.to_numeric(v, errors="coerce")) for v in data2.values()],
        "median": [np.nanmedian(pd.to_numeric(v, errors="coerce")) for v in data2.values()],
        "std": [np.nanstd(pd.to_numeric(v, errors="coerce")) for v in data2.values()],
        "n": [np.count_nonzero(~np.isnan(pd.to_numeric(v, errors="coerce"))) for v in data2.values()]}, index=data2.keys())
    summary2.append(summary)

    if group == 'ALL':
        df = pd.DataFrame(data1)
        df['Subject'] = subj_24hr.iloc[:,0]
        # Select the variables of interest
        cols = ["Unbouted", "Short", "Medium", "Long"]
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

        # Compute correlation matrix (default = Pearson)
        corr_matrix = df[cols].corr().round(4)
        # Write to CSV
        corr_matrix.to_csv(nimbal_drive+paper_path+"Figures_tables\\Tables\\Table_ALL_SML_correlations.csv")

        #Cluster analysis
        subset_cluster = df[cols]

        #sum across subject days
        print ('Run and plot cluster analysis....')
        cluster_data = subset_cluster
        ncluster = 3
        data_out, labels = clustering(cluster_data, ncluster=ncluster)
        subject_clusters = pd.DataFrame({'SUBJECT': cluster_data.index,'GROUP': labels})
        subject_clusters['GROUP'] = subject_clusters['GROUP'].replace({0: 'Low'})
        subject_clusters['GROUP'] = subject_clusters['GROUP'].replace({1: 'High'})
        subject_clusters['GROUP'] = subject_clusters['GROUP'].replace({2: 'High/Low'})
        data_out['cluster'] = data_out['cluster'].replace({0: 'Low'})
        data_out['cluster'] = data_out['cluster'].replace({1: 'High'})
        data_out['cluster'] = data_out['cluster'].replace({2: 'High/Low'})

        x = np.arange(len(cols))
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data_out, x='feature', y='value', hue='cluster', hue_order=['Low','High/Low','High'], palette='Set2', linewidth=5)
        plt.xlabel('Bout durations', fontsize=18)
        plt.ylabel('Strides/day', fontsize=18)
        plt.xticks(ticks=x, labels=cols, fontsize=16)
        plt.yticks(fontsize=18)
        plt.title('Bout pattern clusters', fontsize=24)
        plt.legend(title='Cluster', title_fontsize=18, fontsize=18)
        plt.savefig(nimbal_drive + paper_path + "Figures_tables\\Figures\\Figure_ALL_clusters_n3.png")
        plt.close()

        #sum across subject days
        print ('Run and plot cluster analysis....')
        cluster_data = subset_cluster
        ncluster = 4
        data_out, labels = clustering(cluster_data, ncluster=ncluster)
        subject_clusters = pd.DataFrame({'SUBJECT': cluster_data.index,'GROUP': labels})
        subject_clusters['GROUP'] = subject_clusters['GROUP'].replace({0: '1'})
        subject_clusters['GROUP'] = subject_clusters['GROUP'].replace({1: '2'})
        subject_clusters['GROUP'] = subject_clusters['GROUP'].replace({2: '3'})
        subject_clusters['GROUP'] = subject_clusters['GROUP'].replace({3: '4'})
        data_out['cluster'] = data_out['cluster'].replace({0: '1'})
        data_out['cluster'] = data_out['cluster'].replace({1: '2'})
        data_out['cluster'] = data_out['cluster'].replace({2: '3'})
        data_out['cluster'] = data_out['cluster'].replace({3: '3'})

        x = np.arange(len(cols))
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data_out, x='feature', y='value', hue='cluster', hue_order=['1','2','3','4'], palette='Set2', linewidth=5)
        plt.xlabel('Bout durations', fontsize=18)
        plt.ylabel('Strides/day', fontsize=18)
        plt.xticks(ticks=x, labels=cols, fontsize=16)
        plt.yticks(fontsize=18)
        plt.title('Bout pattern clusters', fontsize=24)
        plt.legend(title='Cluster', title_fontsize=20, fontsize=18)
        plt.savefig(nimbal_drive + paper_path + "Figures_tables\\Figures\\Figure_ALL_clusters_n4.png")
        plt.close()




summary1 = pd.concat(summary1, ignore_index=True)
summary1.to_csv(nimbal_drive+paper_path+"Figures_tables\\Tables\\Table_groups_SML_total_steps.csv")
summary2 = pd.concat(summary2, ignore_index=True)
summary2.to_csv(nimbal_drive+paper_path+"Figures_tables\\Tables\\Table_groups_SML_percent_steps.csv")



'''

if figure6:  #density plot
    #path = nimbal_drive + demo_path
    #demodata = read_demo_ondri_data(path)
    #subj_list = demodata[demodata['COHORT'] == 'Community Dwelling']['SUBJECT']

    #reorder based on meda step totals
    subj_total = subj_24hr.iloc[:,0:2] #selects subj and total median column
    subj_total = subj_total.iloc[1:].reset_index(drop=True)
    sorted = subj_total.sort_values(by=subj_total.columns[1]).reset_index(drop=True)
    subj_list = sorted.iloc[:,0]
    visit = '01'
    window_text = 'win_' + str(window_size) + 's_step_' + str(step_size) + 's_'

    # plot the bout density file
    path_density = nimbal_drive + paper_path + 'Summary_data\\density\\'+study+'\\'
    intensity_blocks = []
    for subj in subj_list:
        file = subj+'_'+visit+'_'+window_text+'_density.csv'
        density_subj = pd.read_csv(path_density + file)
        density_subj = density_subj.iloc[2:].reset_index(drop=True)
        density_subj = density_subj.iloc[:,1:]
        rotated = density_subj.T
        intensity_blocks.append(rotated)
    intensity_matrix = pd.concat(intensity_blocks, axis=0, ignore_index=True)
    intensity_matrix = intensity_matrix.apply(pd.to_numeric, errors='coerce')
    intensity_matrix = intensity_matrix.fillna(0)
    intensity_array = intensity_matrix.to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))
    plt.imshow(intensity_matrix, aspect='auto', cmap='viridis')  # or 'hot', 'plasma', 'magma'
    plt.colorbar(label='Intensity')
    plt.xlabel('Time (minutes) (midnight-midnight)')
    plt.ylabel('Subjects and days stacked')
    plt.title('Daily step density (all subjects/days)')
    plt.tight_layout()
    plt.show()
    print ('pause')

if figure7: #alpha-gini analysis
    #run on control list only first
    subj_summary, subj_fits, all_fit = alpha_gini_bouts(study, root, nimbal_drive, paper_path, subj_lists[0])
    subj_summary['Total bouted'] = plot_24hr_bouted
    subj_summary['Med'] = med
    subj_summary['Long'] = long
    corr_bout_gini = subj_summary.drop(columns=['Subj']).corr()
    print (corr_bout_gini)
    stats_bout_gini = subj_summary.drop(columns=['Subj']).agg(['count', 'mean', 'median', 'std', 'min', 'max']).T
    print (stats_bout_gini)
    if False: #plots powerlaw data
        # ---- Plot ----
        fig = plt.figure(figsize=(8, 6))
        # 1) Plot each subject's **fitted power-law PDF**
        for subj, fit in subj_fits.items():
            fit.power_law.plot_pdf(color='lightgrey', linestyle='-', linewidth=1.5)
        # 2) Plot **pooled empirical** and **pooled fitted** in highlighted style
        # all_fit.plot_pdf(color='black', linewidth=2.5, alpha=0.5, label='Pooled empirical')
        all_fit.power_law.plot_pdf(color='crimson', linestyle='--', linewidth=3, label='Pooled power-law fit')
        # plt.axvline(all_fit.xmin, color='crimson', linestyle=':', linewidth=2,
        #        label=f'pooled xmin={all_fit.xmin:.2g}')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.title('Subject and Group Power-law Fits')
        plt.xlabel('Log - Bout duration (secs)', fontsize=14)
        plt.ylabel('Log - Proportion', fontsize=14)
        # plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

'''