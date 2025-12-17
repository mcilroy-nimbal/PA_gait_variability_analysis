import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from Functions import (wake_sleep, steps_by_day, step_density_sec,
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

#which subjects?
master_subj_list = select_subjects(nimbal_drive, study)
#master_subj_list = ['OND09_SBH0006']
print('\nTotal # subjects: \t' + str(len(master_subj_list)) + '\n')
print('First 5 subject in list...' + str(master_subj_list[:5])+'\n')

#select specific subjects from the study group
#select ONDRI
group_name = ['Control', 'PD', 'ADMCI']

study = 'OND09'
path = nimbal_drive + demo_path
demodata = read_demo_ondri_data(path)
#demogarphci details by group
#AGE,SEX,MRTL_STATUS,EMPLOY_STATUS, LIVING_CIRCUM
#data = demodata
#categ, cont = create_table(data, ['AGE'], ['SEX','MRTL_STATUS','EMPLOY_STATUS', 'LIVING_CIRCUM'])

for i in range(3):
    if i == 0:
        label = 'Community Dwelling'
    elif i == 1:
        label = 'PD'
    elif i == 2:
        label = 'AD\MCI'
    #data = demodata[demodata['COHORT'] == label]
    #categ, cont = create_table(data, ['AGE'], ['SEX','MRTL_STATUS','EMPLOY_STATUS', 'LIVING_CIRCUM'])
    #print (group_name[i], categ, cont)

#subject lists
list1 = demodata[demodata['COHORT'] == 'Community Dwelling']['SUBJECT']
list2 = demodata[demodata['COHORT'] == 'PD']['SUBJECT']
list3 = demodata[demodata['COHORT'] == 'AD/MCI']['SUBJECT']
subj_lists = [list1, list2, list3]


subject_tables = False

create_master_graph = False

create_bins = False
create_density = False
calc_basic_stats = False

tables1 = False
tables2 = False

seven_day = True

plot = True

figure1 = False #swarm totals
figure1b = False

figure2 = False  #KDE distibiton - bouts/unbouted
figure3 = False #bout disitbution - steps and %

figure4 = False #steps per class (s,m,l)
figure4b = False #% steps per class (s,m,l)

figure5 = False # 3 coffecint of varaition - bewteen day

figure6 = True #raw density plot

figure7 = False #gini

central = 'mean'
central1 = 'Mean'
#create summary data files

#######################################################################
#creates files
#groups  - 0 Control, 1 PD, 2 ADMCI
group = 0
if seven_day:
    path = nimbal_drive + demo_path
    path_daily = nimbal_drive + paper_path + 'Summary_data\\OND09_24hr_bout_width_daily_bins_with_unbouted.csv'
    subj_daily = pd.read_csv(path_daily)
    #select only controls
    #keep only 7 days

    #Total steps per bin as percentage of total (not by mean / day)

    #tabulate the mean percentages per bin

    #



if create_master_graph:
    all_bouts_histogram(study, root, nimbal_drive, paper_path, subj_lists[group])

if create_bins:
    time_window ='1010'#'24hr' '1010'  'wake'
    create_bin_files(time_window, study, root, nimbal_drive, paper_path, master_subj_list,
                             bin_list_steps, bin_width_time)

if create_density:
    create_density_files(study, root, nimbal_drive, group_name[group], paper_path, master_subj_list)

if calc_basic_stats:

    window = '24hr'
    step_vs_dur = False
    #calcualte medians and std for each bout and clustred bouts
    calc_basic_stride_bouts_stats(step_vs_dur, nimbal_drive, study, window, paper_path, subj_lists[group], group_name[group])

    step_vs_dur = True #run duraiton file
    calc_basic_stride_bouts_stats(step_vs_dur, nimbal_drive, study, window, paper_path, subj_lists[group], group_name[group])

    #SML_median, SML_std, SML_pct_median, SML_pct_std = bouts_SML(nimbal_drive, study, window, paper_path, subj_list)

#################################################


if plot:
    ##############################################################
    #set data for future plots
    #
    path = nimbal_drive + demo_path

    path_24hr = nimbal_drive + paper_path + 'Summary_data\\' + study + '_24hr_' + group_name[group] + '_bout_duration_'
    subj_24hr = pd.read_csv(path_24hr +'_subj_stats.csv', header=[0, 1], skiprows=[2])

    subj_pct_24hr = pd.read_csv(path_24hr + '_pct_subj_stats.csv', header=[0, 1], skiprows=[2])

    #slects only median or mean based on the preset variables called central
    group_24hr = pd.read_csv(path_24hr +'_group_stats_'+central+'.csv')
    group_pct_24hr = pd.read_csv(path_24hr + '_pct_group_stats_'+central+'.csv')

    # group wide CVS for histogram
    group_24hr_cvs = pd.read_csv(path_24hr +'_group_stats_cv_'+central+'.csv')
    group_pct_24hr_cvs = pd.read_csv(path_24hr + '_pct_group_stats_cv_'+central+'.csv')

    #path_1010 = nimbal_drive + paper_path + 'Summary_data\\' + study + '_1010_' + group_name[group] + '_bout_duration_'
    #subj_1010 = pd.read_csv(path_1010 +'_subj_stats.csv', header=[0, 1] )
    #group_1010 = pd.read_csv(path_1010 + '_group_stats_'+central+'.csv')
    #subj_pct_1010 = pd.read_csv(path_1010 + '_pct_subj_stats.csv', header=[0, 1])
    #group_pct_1010 = pd.read_csv(path_1010 + '_pct_group_stats_'+central+'.csv')

    #path_wake = nimbal_drive + paper_path + 'Summary_data\\' + study + '_wake_' + group_name[group] + '_bout_duration_'
    #subj_wake = pd.read_csv(path_wake + '_subj_stats.csv', header=[0, 1])
    #group_wake = pd.read_csv(path_wake + '_group_stats_'+central+'.csv')
    #subj_pct_wake = pd.read_csv(path_wake + '_pct_subj_stats.csv', header=[0, 1])
    #group_pct_wake = pd.read_csv(path_wake + '_pct_group_stats_'+central+'.csv')

    plot_24hr_all = subj_24hr[('window_total_strides', central)]
    plot_24hr_all = plot_24hr_all[1:]
    #plot_1010_all = subj_1010[('window_total_strides', central)]
    #plot_wake_all = subj_wake[('window_total_strides', central)]

    plot_24hr_unbouted = subj_24hr[('window_not_bouted_strides', central)]
    plot_24hr_unbouted = plot_24hr_unbouted[1:]
    #plot_1010_unbouted = subj_1010[('window_not_bouted_strides', central)]
    #plot_wake_unbouted = subj_wake[('window_not_bouted_strides', central)]

    plot_24hr_bouted = plot_24hr_all - plot_24hr_unbouted
    #plot_1010_bouted = plot_1010_all - plot_1010_unbouted
    #plot_wake_bouted = plot_wake_all - plot_wake_unbouted

    short_24hr_bouted = subj_24hr[[('strides_<_5', central), ('strides_<_10', central),('strides_<_30', central)]]
    med_24hr_bouted = subj_24hr[[('strides_<_60', central), ('strides_<_180', central)]]
    long_24hr_bouted = subj_24hr[[('strides_<_600', central), ('strides_>_600', central)]]

    short = short_24hr_bouted.sum(axis=1)
    med = med_24hr_bouted.sum(axis=1)
    long = long_24hr_bouted.sum(axis=1)


    corr_df = pd.DataFrame({'All': plot_24hr_all, 'Unbouted': plot_24hr_unbouted,'Short': short,'Medium': med,'Long': long})
    corr = corr_df.corr(method='spearman')
    print(corr)

    short_pct = 100*(short / plot_24hr_all)
    med_pct = 100 * (med / plot_24hr_all)
    long_pct = 100 * (long / plot_24hr_all)
    unbouted_pct = 100 * (plot_24hr_unbouted / plot_24hr_all)



    #night_time_totals = plot_24hr_all - plot_wake_all


    if tables1:
        #mean values for table
        # Put them in a dictionary
        data = {"All": plot_24hr_all, "Bouted only": plot_24hr_bouted, "Unbouted only": plot_24hr_unbouted}
        #data = {"All": plot_1010_all, "Bouted only": plot_1010_bouted, "Unbouted only": plot_1010_unbouted}
        #data = {"All": plot_wake_all, "Bouted only": plot_wake_bouted, "Unbouted only": plot_wake_unbouted}

        summary = pd.DataFrame({
        "mean": [np.nanmean(v) for v in data.values()],
        "median": [np.nanmedian(v) for v in data.values()],
        "std": [np.nanstd(v, ddof=1) for v in data.values()],
        "n": [np.count_nonzero(~np.isnan(v)) for v in data.values()]}, index=data.keys())
        print (summary)

    if tables2:
        data = {"Unbouted": plot_24hr_unbouted, "Short": short, "Medium": med, "Long": long}
        summary = pd.DataFrame({
            "mean": [np.nanmean(v) for v in data.values()],
            "median": [np.nanmedian(v) for v in data.values()],
            "std": [np.nanstd(v, ddof=1) for v in data.values()],
            "n": [np.count_nonzero(~np.isnan(v)) for v in data.values()]}, index=data.keys())
        print(summary)
        data = {"Unbouted": unbouted_pct, "Short": short_pct, "Medium": med_pct, "Long": long_pct}
        summary = pd.DataFrame({
            "mean": [np.nanmean(v) for v in data.values()],
            "median": [np.nanmedian(v) for v in data.values()],
            "std": [np.nanstd(v, ddof=1) for v in data.values()],
            "n": [np.count_nonzero(~np.isnan(v)) for v in data.values()]}, index=data.keys())
        print(summary)

    if figure1:
        # Sample DataFrame with two numeric columns
        #df = pd.DataFrame({'All steps': plot_24hr_all, 'Bouted only': plot_24hr_bouted, 'Unbouted': plot_24hr_unbouted})
        #df = pd.DataFrame({'24 HR': plot_24hr_bouted, 'Wake': plot_wake_bouted, '10AM-10PM': plot_1010_bouted})
        df = pd.DataFrame({'24 HR': plot_24hr_all, 'Wake': plot_wake_all, '10AM-10PM': plot_1010_all})
        # Melt the DataFrame to long format for seaborn
        melted_df = df.melt(var_name='Window', value_name='Total steps')

        # Create the swarm plot
        plt.figure(figsize=(6, 5))
        #sns.swarmplot(x='Window', y='Total steps', data=melted_df, size=6)
        #sns.catplot(data= melted_df, x='Window', y='Total steps', kind='swarm',palette={'24 HR': 'skyblue', '10AM-10PM': 'salmon'})

        # Create violin plot
        #sns.violinplot(x='Window', y='Total steps', data=melted_df, inner=None, palette={'24 HR': 'skyblue','Wake': 'magenta', '10AM-10PM': 'salmon'})
        sns.boxplot(data=melted_df, x="Window", y="Total steps", showcaps=True, hue="Window", boxprops={'facecolor': 'None'},  # transparent box so swarm is visible
            showfliers=False) # hide outliers (swarm will show them)

        # Overlay swarm plot
        sns.swarmplot(x='Window', y='Total steps', data=melted_df, hue="Window", size=4)

        plt.ylim(bottom=0, top=12000)
        #plt.title('Steps / day comparing time window')
        plt.xlabel('Analysis Window', fontsize=14)
        #plt.xlabel('Step classification', fontsize=14)

        plt.ylabel('Average unilateral steps / day', fontsize=14)
        plt.tight_layout()
        plt.show()

    if figure1b:
        # Create the swarm plot
        plt.figure(figsize=(8, 6))
        # Create violin plot
        sns.violinplot(y=night_time_totals, inner=None, color='magenta')
        # Overlay swarm plot
        sns.swarmplot(y=night_time_totals, color='black', size=4)

        plt.ylim(bottom=0, top=1000)

        plt.title('Steps / night')
        plt.xlabel('Window')
        plt.ylabel(central + ' unilateral steps / night')
        plt.tight_layout()
        plt.show()

    if figure2:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)

        # Plot 1
        sns.kdeplot(x=plot_24hr_all, fill=True, label='24 HR', color='skyblue', alpha=0.5, ax=axes[0])
        sns.kdeplot(x=plot_wake_all, fill=True, label='Wake', color='magenta', alpha=0.5, ax=axes[0])
        sns.kdeplot(x=plot_1010_all, fill=True, label='10AM–10PM', color='salmon', alpha=0.5, ax=axes[0])
        axes[0].set_title('Total unilateral steps')
        axes[0].set_xlabel('Step Count')
        axes[0].set_yticks([])
        axes[0].legend()

        # Plot 2
        sns.kdeplot(x=plot_24hr_unbouted, fill=True, label='24 HR', color='skyblue', alpha=0.5, ax=axes[1])
        sns.kdeplot(x=plot_wake_unbouted, fill=True, label='Wake', color='magenta', alpha=0.5, ax=axes[1])
        sns.kdeplot(x=plot_1010_unbouted, fill=True, label='10AM–10PM', color='salmon', alpha=0.5, ax=axes[1])
        axes[1].set_title('Unbouted unilateral steps')
        axes[1].set_xlabel('Step Count')
        axes[1].set_yticks([])
        axes[1].legend()

        # Plot 3
        sns.kdeplot(x=plot_24hr_bouted, fill=True, label='24 HR', color='skyblue', alpha=0.5, ax=axes[2])
        sns.kdeplot(x=plot_wake_bouted, fill=True, label='Wake', color='magenta', alpha=0.5, ax=axes[2])
        sns.kdeplot(x=plot_1010_bouted, fill=True, label='10AM–10PM', color='salmon', alpha=0.5, ax=axes[2])
        axes[2].set_title('Bouted unilateral steps')
        axes[2].set_xlabel('Step Count')
        axes[2].set_yticks([])
        axes[2].legend()

        # Final layout
        fig.suptitle('Step Count Distributions Across Time Windows', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

    if figure3:

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
        plt.show()

    if figure4:
        df = pd.DataFrame({'Unbouted': unbouted_pct, 'Short': short_pct, 'Medium': med_pct, "Long": long_pct})
        melted_df = df.melt(var_name='Bout class', value_name='% Total steps')

        # Create the swarm plot
        plt.figure(figsize=(6, 5))
        # sns.swarmplot(x='Window', y='Total steps', data=melted_df, size=6)
        # sns.catplot(data= melted_df, x='Window', y='Total steps', kind='swarm',palette={'24 HR': 'skyblue', '10AM-10PM': 'salmon'})

        # Create violin plot
        # sns.violinplot(x='Window', y='Total steps', data=melted_df, inner=None, palette={'24 HR': 'skyblue','Wake': 'magenta', '10AM-10PM': 'salmon'})
        sns.boxplot(data=melted_df, x="Bout class", y="% Total steps", showcaps=True, hue="Bout class",
                    boxprops={'facecolor': 'None'},  # transparent box so swarm is visible
                    showfliers=False)  # hide outliers (swarm will show them)

        # Overlay swarm plot
        sns.swarmplot(x='Bout class', y='% Total steps', data=melted_df, hue="Bout class",
                      palette=["red", "magenta", "blue", "green"], size=4)

        plt.ylim(bottom=0)
        # plt.title('Steps / day comparing time window')
        plt.xlabel('Bout classification', fontsize=14)
        # plt.xlabel('Step classification', fontsize=14)

        plt.ylabel('% of total average unilateral steps / day', fontsize=14)
        plt.tight_layout()
        plt.show()


    if figure4b:
        #plot_labels = ['Total','Unbouted', '<5', '5-10', '10-30', '30-60', '60-180', '180-600', '>600']

        fig, axs = plt.subplots(2,2, figsize=(8, 6), sharex=True, sharey=True)

        #sns.scatterplot(x=plot_24hr_all, y=plot_24hr_unbouted, color='red', label='Unbouted', ax=axs[0,0])
        #sns.scatterplot(x=plot_24hr_all, y=short, color='magenta', label='Short <30 sec', ax=axs[0,1])
        #sns.scatterplot(x=plot_24hr_all, y=med, color='blue', label='Medium 30-180 sec', ax=axs[1,0])
        #sns.scatterplot(x=plot_24hr_all, y=long, color='green', label='Long  > 180 sec',ax=axs[1,1])

        sns.regplot(x=plot_24hr_all, y=plot_24hr_unbouted, color='red', label='Unbouted',
                    scatter_kws={"s": 20, "alpha": 0.7}, line_kws={"color": "grey"}, ci=None, ax=axs[0,0])
        sns.regplot(x=plot_24hr_all, y=short, color='magenta', label='Short <30 sec',
                    scatter_kws={"s": 20, "alpha": 0.7}, line_kws={"color": "grey"}, ci=None, ax=axs[0, 1])
        sns.regplot(x=plot_24hr_all, y=med, color='blue', label='Medium 30-180 sec',
                    scatter_kws={"s": 20, "alpha": 0.7}, line_kws={"color": "grey"}, ci=None, ax=axs[1, 0])
        sns.regplot(x=plot_24hr_all, y=long, color='green', label='Long > 180 sec',
                    scatter_kws={"s": 20, "alpha": 0.7}, line_kws={"color": "grey"}, ci=None, ax=axs[1, 1])

        plt.suptitle('Total steps versus bouted and unbouted steps')

        axs[0,0].set_ylabel(central + ' daily steps')
        axs[1,0].set_ylabel(central + ' daily steps')
        axs[0,0].set_xlabel('')
        axs[0,1].set_xlabel('')
        axs[1,0].set_xlabel('Total steps ('+central+') / day')
        axs[1,1].set_xlabel('Total steps ('+central+') / day')
        axs[0,0].set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

    if figure5:

        #plot_labels = ['Total', 'Unbouted', '<5', '5-10', '10-25', '25-50', '50-100', '100-300', '>300']
        plot_labels = ['Unbouted', '<5', '5-10', '10-30', '30-60', '60-180', '180-600', '>600']

        central_24hr_nototal = group_24hr_cvs.iloc[1:].reset_index(drop=True)
        central_24hr_pct_nototal = group_pct_24hr_cvs.iloc[1:].reset_index(drop=True)

        fig, axs = plt.subplots(2, figsize=(6, 9))
        # median std strides
        ticks = list(range(len(plot_labels)))
        axs[0].bar(central_24hr_nototal.index, central_24hr_nototal[central1].values, yerr=central_24hr_nototal['Std'], capsize=5, color='lightblue', edgecolor='black')
        #axs[0].set_title(central1 + ' between day variation by bout length')
        axs[0].set_xlabel('Bout length (sec)', fontsize=14)
        axs[0].set_ylabel('Coefficient of variation', fontsize=14)
        axs[0].set_xticks(ticks=ticks, labels=plot_labels, fontsize=12)
        axs[0].set_ylim(bottom=0)

        # median std strides
        ticks = list(range(len(plot_labels)))
        axs[1].bar(central_24hr_pct_nototal.index, central_24hr_pct_nototal[central1].values, yerr=central_24hr_pct_nototal['Std'], capsize=5, color='lightgreen', edgecolor='black')
        #axs[1].set_title(central1 + ' between day variations in percentage by bout length')
        axs[1].set_xlabel('Bout length (sec)', fontsize=14)
        axs[1].set_ylabel('Coefficient ov variation', fontsize=14)
        axs[1].set_xticks(ticks=ticks, labels=plot_labels, fontsize=12)
        axs[1].set_ylim(bottom=0)

        plt.tight_layout()
        plt.show()

    if figure6:
        #path = nimbal_drive + demo_path
        #demodata = read_demo_ondri_data(path)
        #subj_list = demodata[demodata['COHORT'] == 'Community Dwelling']['SUBJECT']

        #reorder based on meda step totals
        subj_total = subj_24hr.iloc[:,0:2] #selects subj and total median column
        subj_total = subj_total.iloc[1:].reset_index(drop=True)
        sorted = subj_total.sort_values(by=subj_total.columns[1]).reset_index(drop=True)
        subj_list = sorted.iloc[:,0]
        visit = '01'

        # plot the bout density file
        path_density = nimbal_drive + paper_path + 'Summary_data\\density\\'+study+'\\'
        intensity_blocks = []
        for subj in subj_list:
            file = subj+'_'+visit+'_60sec_density.csv'
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
