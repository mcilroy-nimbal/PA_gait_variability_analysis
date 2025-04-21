#THIS CODE: analysis the data produced in STEP 1
#Bins file
#Density files

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from Functions import wake_sleep, bout_bins, steps_by_day, step_density_1min,read_orig_clean_demo
import numpy as np
import seaborn as sns
import datetime
import openpyxl

#set up paths
root = 'W:'
nimbal_drive = 'O:'
paper_path =  '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'
log_out_path = nimbal_drive + paper_path + 'Log_files\\'
summary_path = nimbal_drive + paper_path + 'Summary_data\\'


###########################################
#read in the cleaned data file for the HANNDS methods paper
nimbal_dr = 'o:'
new_path = '\\Papers_NEW_April9\\Shared_Common_data\\OND09\\'
#Import data files - use this if file already created
demodata = pd.read_csv(nimbal_dr+new_path+"OND09_ALL_01_CLIN_DEMOG_2025_CLEAN_HANDDS_METHODS_N245.csv")
demodata['SUBJECT'] = demodata['SUBJECT'].astype(str).str.replace('_', '')
#merge dual diagonis - other MCI
demodata['COHORT'] = demodata['COHORT'].replace('MCI;CVD','CVD')
demodata['COHORT'] = demodata['COHORT'].replace('MIC;PD','PD')
demodata['COHORT'] = demodata['COHORT'].replace('AD;MCI','MCI')
#collapse AD MCI
demodata['COHORT'] = demodata['COHORT'].replace('AD','MCI')
demodata['COHORT'] = demodata['COHORT'].replace('MCI','AD/MCI')

#nimbaldata$ / OND09(HANDDS - ONT) / HANDDS methods paper data / HANDDS Methods Paper - Data Cleaning and Analysis Methods.docx
#persons self report vereus clincian - so we need to be careful

#select subset
#loop through disease cohorts
#demodata['COHORT']
#demodata['AGE']
#demodata['SEX']
#demodata['MRTL_STATUS']

#reads bout file as well
bouts_all = pd.read_csv(summary_path + 'steps_daily_bins.csv')
#cleans up the subj id removes _
bouts_all['subj'] = bouts_all['subj'].astype(str).str.replace('_', '')

#selects subset
#by step totals
#bouts = bouts_all[(bouts_all['total'] / bouts_all['ndays']) > 5000]

#by demographic
#subset = demodata[(demodata['COHORT'] != 'Community Dwelling')]
#subset = demodata[(demodata['AGE'] > 55) & (demodata['AGE'] < 75)]


gait_bout = False
write_density = False
plot_density_summary = False
plot_density_raw = False
plot_demo = False
run_variability_bouts = True

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
    print()

if write_density:
    # bout density data
    # create the bout density summary file
    data_path = summary_path+'density\\'
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    counter = 0
    bin_width = 5
    # Compute bin edges
    bins = np.arange(start=-bin_width, stop=100, step=bin_width)
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    columns = ['subj','day'] + bin_labels
    hist = pd.DataFrame(columns=columns)
    step = -50
    for index, file in enumerate(files):
        #select subj name and visit name for writing to file
        parts = file.split('_')
        subj = parts[0]+parts[1]
        visit = parts[2]
        print(f'\rSubj #: {index}' + ' of ' + str(len(files)), end='', flush=True)
        subj_density = pd.read_csv(data_path+file)
        del subj_density[subj_density.columns[0]]

        # Create histogram
        for day, col in enumerate(subj_density.columns):
            # Bin the data
            binned = pd.cut(subj_density[col], bins=bins, right = True)
            # Count the number of observations in each bin
            counts = binned.value_counts().sort_index()
            row = counts.values
            row1 = [subj, day+1] + row.tolist()
            hist.loc[len(hist)] = row1
    hist.to_csv(summary_path+'freq_step_per_min.csv')


if plot_density_summary:
    # plot the bout density file
    data_path = summary_path+'freq_step_per_min.csv'
    hist_all = pd.read_csv(data_path)
    #delete column 0 - lft over index?
    hist_all = hist_all.drop(hist_all.columns[0], axis=1)
    # salects subset
    hist = hist_all[hist_all['subj'].isin(subset['SUBJECT'])]
    #hist_ = hist_all

    #only plot bin columns - inore subjct and day
    cols_plot = hist.columns[3:]

    for idx, row in hist.iterrows():
            vals = row[cols_plot].values
            x = cols_plot
            plt.plot(x, vals)

    plt.show()
    print ()

if plot_density_raw:
    # bout density data
    # plot density summary file
    data_path = summary_path+'density\\'
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    count = 0
    fig, ax = plt.subplots(figsize=(10,6))
    time = np.linspace(1, 1440, 1440)

    for index, file in enumerate(files):
        #print(f'\rSubj #: {index}' + ' of ' + str(len(files)), end='', flush=True)
        parts = file.split('_')
        subj = parts[0] + parts[1]
        #cohort = demodata.loc[demodata['SUBJECT'] == subj, 'COHORT'].values[0]
        #age = demodata.loc[demodata['SUBJECT'] == subj, 'AGE'].values[0]

        sub_set = bouts_all[bouts_all['subj'] == subj]
        n_days = len(sub_set)
        tot_steps = sub_set['total'].sum()
        tot_steps_day = tot_steps / n_days

        if (tot_steps_day > 0) and (tot_steps_day<2001):
            print(f'\rSubj #: {index}' + ' of ' + str(len(files)), end='', flush=True)
            subj_density = pd.read_csv(data_path+file)
            del subj_density[subj_density.columns[0]]

            # plot by day and subject
            for day, col in enumerate(subj_density.columns):
                data1 = subj_density[col].values
                data2 = data1.reshape(1, -1)
                ax.imshow(data2, aspect = 'auto', cmap='viridis', interpolation=None, extent=[time[0], time[-1], count, count+1])
                #print(f'\rCount #: {count}', end='', flush=True)
                count=count+1
            count=count+1
    ax.set_ylim(0, count)
    ax.set_ylabel("Subjects - and days within subjects")
    ax.set_xlabel("Time (mins/day")
    #ax.set_title("Step Density time series - Subject: "+ subj+"  Cohort: "+cohort+"  Age: "+str(age))
    ax.set_title("Step Density time series - aLL  < 2000 steps/day")
    plt.colorbar(ax.images[0], ax=ax, label='Step density (steps/minute)')  # colorbar from the first image
    plt.tight_layout()
    plt.show()
    #plt.savefig(data_path+'images\\'+subj+'_density_days.pdf')
print()

if run_variability_bouts:

