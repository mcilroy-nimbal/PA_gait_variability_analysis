#THIS CODE: analysis the data produced in STEP 1
#Bins file
#Density files

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plot_tabulate_functions import plot_density_raw
import numpy as np
import seaborn as sns
import datetime
import openpyxl


def coeff_var(x):
    return np.std(x, ddof=1) / np.mean(x) if np.mean(x) != 0 else np.nan

def calc_basic_stride_bouts_stats(step_vs_dur, nimbal_drive, study, window, path, subject_list, group_name):
    #stride totals - by bouts
    #averages across days for each subject
    if step_vs_dur:
        steps = pd.read_csv(nimbal_drive + path + 'Summary_data\\' + study + '_' + window + '_bout_steps_daily_bins.csv')
    else:
        steps = pd.read_csv(nimbal_drive + path + 'Summary_data\\' + study + '_' + window + '_bout_width_daily_bins.csv')

    if window == 'wake':
        #remove any less than 10 hours - for cocnern of code right now
        # Extract the number from the string and convert to integer
        steps['hours'] = steps['window'].str.extract(r'_(\d+)hr')
        steps = steps[steps['hours'].notna()]  # Keep only rows with a number
        steps['hours'] = steps['hours'].astype(int)
        steps = steps[steps['hours'] > 10]
        steps.drop(columns=['hours'], inplace=True)

    #select only specific subjects
    steps = steps[steps['subj'].isin(subject_list)]

    #select column headers to select only stride columns and
    # these are the strides per bout class
    stride_bouts = steps.columns[steps.columns.str.startswith('strides_')].tolist()

    # this adds unbouted
    stride_bouts.insert(0, 'window_not_bouted_strides')
    stride_bouts.insert(0, 'window_total_strides')

    # calculate the percentage of steps in bouts relative to total (daily)
    for col in stride_bouts:
        steps[col + '_pct'] = steps[col] / steps['window_total_strides'] * 100
    # these are the prct strides per bout class
    pct_bouts = steps.columns[steps.columns.str.contains('_pct')].tolist()

    # mean bouts setp #s absolute
    #nstride_subj_stats = steps.groupby('subj')[stride_bouts].agg(['mean', 'median', 'std', 'count'])

    # Apply aggregation including CV
    nstride_subj_stats = steps.groupby('subj')[stride_bouts].agg(['mean', 'median', 'std', 'count', coeff_var])
    #rename the CV column for clarity
    nstride_subj_stats = nstride_subj_stats.rename(columns={'coeff_var': 'cv'})

    # Extract median and std columns using .xs()
    medians = nstride_subj_stats.xs('median', axis=1, level=1)
    cvs = nstride_subj_stats.xs('cv', axis=1, level=1)

    # Calculate mean and std
    #nstride_group_stats = pd.DataFrame({'Median': medians.median(),'Std': medians.std(),'N' : medians.count() })
    nstride_group_stats = pd.DataFrame({'Median': medians.median(),'Std': medians.std(), 'N': medians.count()})
    nstride_group_stats_cvs = pd.DataFrame({'Median': cvs.median(), 'Std': cvs.std(), 'N': cvs.count()})

    # mean bouts setp #s absolute
    #nstride_pct_subj_stats = steps.groupby('subj')[pct_bouts].agg(['mean', 'median', 'std', 'count'])
    # Apply aggregation including CV
    nstride_pct_subj_stats = steps.groupby('subj')[pct_bouts].agg(['mean', 'median', 'std', 'count', coeff_var])
    # rename the CV column for clarity
    nstride_pct_subj_stats = nstride_pct_subj_stats.rename(columns={'coeff_var': 'cv'})

    # Extract median and std columns using .xs()
    medians = nstride_pct_subj_stats.xs('median', axis=1, level=1)
    cvs = nstride_pct_subj_stats.xs('cv', axis=1, level=1)

    # Calculate mean and std
    #nstride_pct_group_stats = pd.DataFrame({'Median': medians.median(), 'Std': medians.std(), 'N' : medians.count()})
    nstride_pct_group_stats = pd.DataFrame({'Median': medians.median(), 'Std': medians.std(), 'N': medians.count()})
    nstride_pct_group_stats_cvs = pd.DataFrame({'Median': cvs.median(), 'Std': cvs.std(), 'N': cvs.count()})


    full_path = nimbal_drive + path + 'Summary_data\\' + study + '_' + window + '_' + group_name + '_'
    if step_vs_dur:
        full_path = full_path + 'bout_steps_'
    else:
        full_path = full_path + 'bout_duration_'
    nstride_subj_stats.to_csv(full_path + '_subj_stats.csv', float_format='%.2f')

    nstride_group_stats.to_csv(full_path + '_group_stats.csv',float_format='%.2f' )
    nstride_group_stats_cvs.to_csv(full_path + '_group_stats_cvs.csv', float_format='%.4f')

    nstride_pct_subj_stats.to_csv(full_path + '_pct_subj_stats.csv', float_format='%.2f')

    nstride_pct_group_stats.to_csv(full_path + '_pct_group_stats.csv', float_format='%.2f')
    nstride_pct_group_stats_cvs.to_csv(full_path + '_pct_group_stats_cvs.csv', float_format='%.4f')
    return

def bouts_SML (nimbal_drive, study, window, path, subject_list):
    # stride totals - by bouts
    # averages across days for each subject
    steps = pd.read_csv(nimbal_drive + path + 'Summary_data\\' + study + '_' + window + '_bout_steps_daily_bins_with_unbouted.csv')
    # select only specific subjects
    steps = steps[steps['subj'].isin(subject_list)]

    # select column headers to select only stride columns and
    # these are the strides per bout class
    stride_bouts = steps.columns[steps.columns.str.startswith('strides_')].tolist()

    # creat subset of bouts
    steps['strides_short'] = steps['strides_<_5'] + steps['strides_<_10']
    steps['strides_medium'] = steps['strides_<_25'] + steps['strides_<_50']
    steps['strides_long'] = steps['strides_<_100'] + steps['strides_<_300'] + steps['strides_>_300']
    SML_bouts = ['strides_short','strides_medium','strides_long']
    # this adds unbouted
    SML_bouts.insert(0, 'window_not_bouted_strides')
    SML_bouts.insert(0, 'window_total_strides')

    # mean bouts setp #s absolute
    SML_subj_median = steps.groupby('subj')[SML_bouts].median()
    SML_median = SML_subj_median.median()

    SML_subj_means = steps.groupby('subj')[SML_bouts].mean()
    SML_std = SML_subj_means.std()

    # calculate the percentage of steps in bouts relative to total (daily)
    for col in stride_bouts:
        steps[col + '_pct'] = steps[col] / steps['window_total_strides'] * 100
    # these are the prct strides per bout class
    pct_bouts = steps.columns[steps.columns.str.contains('_pct')].tolist()
    SML_pct_bouts = ['strides_short_pct', 'strides_medium_pct', 'strides_long_pct']
    SML_pct_bouts.insert(0, 'window_not_bouted_strides')
    SML_pct_bouts.insert(0, 'window_total_strides')

    # bouts setp #s percentage
    SML_pct_subj_median = steps.groupby('subj')[SML_pct_bouts].median()
    SML_pct_median = SML_pct_subj_median.median()

    SML_pct_subj_mean = steps.groupby('subj')[SML_pct_bouts].mean()
    SML_pct_std = SML_pct_subj_mean.std()

    return SML_median, SML_std, SML_pct_median, SML_pct_std



    fig, axs = plt.subplots(2, figsize=(8, 9))
    plot_labels = ['Total', 'Unbouted','Short', 'Medium', 'Long']

    # median std strides
    median = SML_median
    std = SML_std
    ticks = list(range(len(plot_labels)))
    axs[0].bar(median.index, median.values, yerr=std.values, capsize=5, color='lightblue', edgecolor='black')
    axs[0].set_title('Median unilateral steps / day')
    axs[0].set_xlabel('Bout length (# unilateral steps)')
    axs[0].set_ylabel('Unilateral steps / day')
    axs[0].set_xticks(ticks=ticks, labels=plot_labels)

    median = SML_pct_median
    std = SML_pct_std
    ticks = list(range(len(plot_labels)))
    axs[1].bar(median.index, median.values, yerr=std.values, capsize=5, color='violet', edgecolor='black')
    axs[1].set_title('Median unilateral steps / day - % of total')
    axs[1].set_xlabel('Bout length (# unilateral steps)')
    axs[1].set_ylabel('Unilateral steps / day')
    axs[1].set_xticks(ticks=ticks, labels=plot_labels)
    plt.tight_layout()
    plt.show()

    return

'''
###########################################
#read in the cleaned data file for the HANNDS methods paper
nimbal_dr = 'o:'
new_path = '\\Papers_NEW_April9\\Shared_Common_data\\OND09\\'

study ='SA-PR01'
demodata = read_demo_data(study)


#reads bout file as well
if study == 'SA-PR01':
    file = 'SA_steps_daily_bins_with_unbouted.csv'
else:
    file = 'steps_daily_bins.csv'
bouts_all = pd.read_csv(summary_path + file)

#cleans up the subj id removes _
if study != 'SA-PR01':
    bouts_all['subj'] = bouts_all['subj'].astype(str).str.replace('_', '')

#select files
data_path = summary_path+'density\\'
temp = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
temp = [f for f in temp if '60sec' in f]
files = [f for f in temp if 'PR01' in f]
select_list=[]
for file in files:
    parts = file.split('_')
    if study == 'SA-PR01':
        var='_'
    else:
        var=''
    subj = parts[0] +var+ parts[1]

    #cohort = demodata.loc[demodata['SUBJECT'] == subj, 'COHORT'].values[0]
    #age = demodata.loc[demodata['SUBJECT'] == subj, 'AGE'].values[0]

    sub_set = bouts_all[bouts_all['subj'] == subj]
    n_days = len(sub_set)
    tot_steps = sub_set['total'].sum()
    tot_steps_day = tot_steps / n_days

    if (tot_steps_day > 5000) and (n_days > 6):
       select_list.append(file)


plot_density_raw(summary_path, select_list, bouts_all, demodata)




#selects subset
#by step totals
#bouts = bouts_all[(bouts_all['total'] / bouts_all['ndays']) > 5000]

#by demographic
#subset = demodata[(demodata['COHORT'] != 'Community Dwelling')]
#subset = demodata[(demodata['AGE'] > 55) & (demodata['AGE'] < 75)]


gait_bout = False
write_density = True
plot_density_summary = False

plot_demo = False
run_variability_bouts = False

if plot_demo:
    bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    cohorts = demodata['COHORT'].unique()
    # Plot KDE curves
    plt.figure(figsize=(8, 5))
    #colors = ['blue', 'orange']
    for i, cohort in enumerate(cohorts):
        subset = demodata[demodata['COHORT'] == cohort]
        sns.kdeplot(
            subset['AGE'],
            label=f'Cohort {cohort}',
            #color=colors[i],
            fill=True,   # Optional: fill under curve
            alpha=0.3,   # Transparency for fill
            linewidth=2
        )
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.title('Smoothed Age Distribution by Disease Type (KDE)')
    plt.legend()
    plt.show()

    plt.hist(demodata['AGE'], bins=10, edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Distribution')
    plt.show()


if gait_bout:
    #Gait bout bin analysis
    #read summary bin file
    #bouts_all = pd.read_csv(summary_path + 'steps_daily_bins.csv')
    #cleans up the subj id removes _
    #bouts_all['subj'] = bouts_all['subj'].astype(str).str.replace('_', '')

    #salects subset
    bouts = bouts_all[bouts_all['subj'].isin(subset['SUBJECT'])]
    #bouts = bouts_all

    use_sums = False
    use_cv = True

    sum1 = []
    for subj in bouts['subj'].unique():
        sub_set = bouts[bouts['subj'] == subj]
        n_days = len(sub_set)
        if use_sums:
            #sums
            tot_steps = sub_set['total'].sum()
            tot_steps_day = tot_steps / n_days
            #original
            unbouted = sub_set['<_3'].sum()
            short = sub_set['<_5'].sum() + sub_set['<_10'].sum()
            medium = sub_set['<_20'].sum() + sub_set['<_50'].sum()
            long = sub_set['<_100'].sum() + sub_set['<_300'].sum() + sub_set['>_300'].sum()
        elif use_cv:
            tot_steps = sub_set['total'].sum()
            tot_steps_day = tot_steps / n_days

            unbouted = sub_set['<_3'].std()/sub_set['<_3'].mean()

            sub_set['short'] = sub_set['<_3']+sub_set['<_5'] + sub_set['<_10']
            short = sub_set['short'].std() / sub_set['short'].mean()

            sub_set['medium'] = sub_set['<_20'] + sub_set['<_50']
            medium = sub_set['medium'].std() / sub_set['medium'].mean()

            sub_set['long'] = sub_set['<_100'] + sub_set['<_300'] + sub_set['>_300']
            long = sub_set['long'].std() / sub_set['long'].mean()

        row = {'subj': subj, 'ndays':n_days, 'unbout': unbouted, 'short': short,
              'medium': medium, 'long': long, 'total':tot_steps}
        sum1.append(row)

    sum_bouts = pd.DataFrame(sum1, columns=['subj','ndays','unbout','short','medium','long','total'])

    #orig plot
    x = sum_bouts['total']/sum_bouts['ndays']
    #denom = sum_bouts['total']
    #denom = sum_bouts['ndays']
    denom = 1
    unbout = sum_bouts['unbout']/denom
    short = sum_bouts['short']/denom
    med = sum_bouts['medium']/denom
    long = sum_bouts['long']/denom

    #plt.scatter(x, unbout, color='red', label = 'unbouted')
    plt.scatter(x, short, color='orange', label='short <10')
    #plt.scatter(x, med, color='blue', label='med 10-50')
    plt.scatter(x, long, color='green', label='long >50')

    #sns.jointplot(data=sum_bouts,x=(sum_bouts['total']/sum_bouts['ndays']), y=unbout, kind="scatter", marginal_kws=dict(bins=20, fill=True))
    #sns.jointplot(x=x, y=short, kind="scatter", marginal_kws=dict(bins=20, fill=True))

    #plt.scatter(long, med, color='blue', label='Med vs Long')
    #plt.scatter(long, short, color='orange', label='Short vs Long')
    #plt.scatter(long, unbout, color='red', label='Unbout vs Long')

    plt.legend()
    plt.title('All  ')
    plt.xlabel('Mean steps / day')
    plt.ylabel('Coefficient of variation')
    plt.xlim(left=0, right=18000)
    plt.ylim(bottom=0, top=1)
    plt.show()
    print()'''

