import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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

##SET up and select subjects

#set up file paths
study = 'OND09'
root = 'W:'
nimbal_drive ='O:'
paper_path = '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'
demo_path = '\\Papers_NEW_April9\\Shared_Common_data\\OND09\\'
visit = "01"

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
group_name = ['Community Dwelling', 'PD', 'AD/MCI', 'CVD']

study = 'OND09'
path = nimbal_drive + demo_path
demodata = read_demo_ondri_data(path)
subject_cohort = demodata[['SUBJECT','COHORT']]
counts = subject_cohort.groupby("COHORT").size().reset_index(name="n")
print(counts)
print('\nTotal # subjects in starting list: \t' + str(len(subject_cohort)) + '\n')

#keep on thos in the groups of interest
selected = subject_cohort[subject_cohort["COHORT"].isin(group_name)]
print('\nTotal # subjects in all target groups list: \t' + str(len(selected)) + '\n')

#STEP 1 - subject # list to include
steps = pd.read_csv(nimbal_drive + paper_path + 'Created_data\\bout_bins\\daily_values\\' + study + '_24hr_bout_width_daily_bins_with_unbouted.csv')

#select only specific subjects
steps = steps[steps['subj'].isin(selected['SUBJECT'])]
steps['Cohort'] = selected['COHORT']

#loop through each subject to see if meets min days
# Step 1: get total rows per subject
counts = steps.groupby('subj')['subj'].transform('size')
# Step 2: keep only valid subjects
steps = steps[(counts >= min_days)]
# Step 3: cap rows per subject at max_days
steps = steps[steps.groupby('subj').cumcount() < max_days]

#need to remove subject SBHY0202 they we in a wheelchair
steps = steps[steps['subj'] != 'OND09_SBH0202']
subject_cohort = subject_cohort[subject_cohort["SUBJECT"].isin(steps["subj"])]
counts = subject_cohort.groupby("COHORT").size().reset_index(name="n")
print(counts)
print('\nTotal # subjects in starting list: \t' + str(len(subject_cohort)) + '\n')

#subject lists
group_lists = [0,1,2,3]
group_lists[0] = subject_cohort[subject_cohort['COHORT'] == 'Community Dwelling']['SUBJECT']
print ('Number subjects CONTROL: '+str(len(group_lists[0])))
group_lists[1] = subject_cohort[subject_cohort['COHORT'] == 'PD']['SUBJECT']
print('Number subjects in PD: '+str(len(group_lists[1])))
group_lists[2] = subject_cohort[subject_cohort['COHORT'] == 'AD/MCI']['SUBJECT']
print('Number subjects in ADMCI: '+str(len(group_lists[2])))
group_lists[3] = subject_cohort[subject_cohort['COHORT'] == 'CVD']['SUBJECT']
print('Number subjects in CVD: '+str(len(group_lists[3])))
combined = pd.concat([group_lists[0],group_lists[1],group_lists[2],group_lists[3]], ignore_index=True)
group_lists.append(combined)

groups = ['Control', 'PD', 'ADMCI', 'CVD', 'ALL']


##################################
#Density analysis

#values for density analysis
window_size = 15
step_size = 1

window_text = 'win_' + str(window_size) + 's_step_' + str(step_size) + 's_'
print('\tWindow size (sec): '+ '\t' + str(window_size))
print('\tOverlap (sec): '+ '\t' + str(step_size) )

#need to run this first to create the density files for each subject in the master list
#creates both density (per unit time) and stride time files for each day and subject
#only need to do this once
create_density = False
group = 4
subjects = group_lists[group] #process ALL
if create_density:
    create_density_files(study, root, nimbal_drive, groups[group], paper_path, subjects,
                     window_size, step_size)



analyze_stride_time = False
calc_basic_stats = False
density_graph = False
compare_density = False
stride_time = False
plot_stride_time = False  #done within stride-time
plot_density2 = False


#create the files
#if create_bout_cadence:
#    min_bout_length = 30
#    #create_cadence_bout_summary

if density_graph:  # density plot
    for index, group in enumerate(groups):
        subjects = group_lists[index]

        #sort subject order based on some feature liek total steps for graphin
        #sorting the subject by mean step count
        #read the bouts data to get step totals for the analysis
        path = nimbal_drive + demo_path
        path_24hr = nimbal_drive + paper_path + 'Summary_data\\' + study + '_24hr_' + group_name[index] + '_bout_duration_'
        subj_24hr = pd.read_csv(path_24hr +'_subj_stats.csv', header=[0, 1], skiprows=[2])
        subj_pct_24hr = pd.read_csv(path_24hr + '_pct_subj_stats.csv', header=[0, 1], skiprows=[2])

        # reorder based on meda step totals set subj list from steps files?
        subj_total = subj_24hr.iloc[:, 0:2]  # selects subj and total median column
        subj_total = subj_total.iloc[1:].reset_index(drop=True)
        sorted = subj_total.sort_values(by=subj_total.columns[1]).reset_index(drop=True)
        subj_list = sorted.iloc[:, 0]
        print('Total # subjects for this analysis: \t' + str(len(subj_list)))

        # plot the bout density file
        path_density = nimbal_drive + paper_path + 'Summary_data\\density\\' + study + '\\'
        intensity_blocks = []
        for subj in subj_list:
            file = subj + '_' + visit + '_' + window_text + '_density.csv'
            raw_data = pd.read_csv(path_density + file)
            density_subj = raw_data.iloc[2:].reset_index(drop=True)
            density_subj = density_subj.iloc[:, 1:]
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
        print('pause')

if calc_preferred:
    max = 3.0
    min = 0.5
    # calc and plot / file pre_density
    path_density = nimbal_drive + paper_path + 'created_data\\density\\' + study + '\\'
    group_stats = []
    for subj in subj_list:
        print('\tSubject: \t'+subj)
        file = subj + '_' + visit + '_' + window_text + '_density.csv'
        raw_data = pd.read_csv(path_density + file)
        raw_data = raw_data.iloc[2:].reset_index(drop=True)
        raw_data = raw_data.iloc[:, 1:]

        #convert to density  strides / sec
        density_subj = raw_data / window_size

        # flatten to 1 array
        combined = density_subj.stack()
        #drop NA , 0 and values >
        #convert na to zeros
        combined = combined.fillna(0)

        #cleaned = combined[(combined.notna()) & (combined != 0)]
        #cropped around min and max to narrow preferred calcualtion
        cleaned = combined
        cropped = cleaned[(cleaned > 0) & (cleaned < max)]

        #stats = cropped.describe()

        # convert to a row and label it with subject ID
        #stats_df = stats.to_frame().T
        #stats_df["subject"] = subj
        #group_stats.append(stats_df)

        #caluclate peak value from histogram need bin #s
        #counts, bins = np.histogram(cropped, bins=50)
        #peak_bin_index = counts.argmax()
        #peak_bin_start = bins[peak_bin_index]
        #peak_bin_end = bins[peak_bin_index + 1]
        #print("\tPeak bin:\t", peak_bin_start, "to", peak_bin_end)

        if cropped.dropna().empty:
            print(f"No data for {subj}, skipping")
            continue

        #calculate peak based on KDE
        kde = gaussian_kde(cropped)
        # Evaluate KDE on a grid
        x_grid = np.linspace(cropped.min(), cropped.max(), 1000)
        density = kde(x_grid)
        plt.plot(x_grid, density)
        #Peak = x where density is highest
        #peak_value = x_grid[np.argmax(density)]
        #print("\tKDE peak:\t", peak_value)
        #print ('done')


        '''
        plt.figure(figsize=(10, 6))
        plt.imshow(intensity_matrix, aspect='auto', cmap='viridis')  # or 'hot', 'plasma', 'magma'
        plt.colorbar(label='Intensity')
        plt.xlabel('Time (minutes) (midnight-midnight)')
        plt.ylabel('Subjects and days stacked')
        plt.title('Daily step density (all subjects/days)')
        plt.tight_layout()
        plt.show()
        print('pause')
        '''

    plt.xlabel("Stride Density")
    plt.ylabel("Density")
    plt.title("Distribution Density by Subject")
    plt.legend()
    plt.grid(True)
    plt.show()

    #final_stats = pd.concat(group_stats, ignore_index=True)
    #path_density = nimbal_drive + paper_path + 'Summary_data\\density\\' + study + '\\'
    #file = 'All_' + visit + '_' + window_text + '_cropped_density_stats.csv'
    #final_stats.to_csv(path_density+file, index=False)
    #print (final_stats)

if compare_density:
    path_density = nimbal_drive + paper_path + 'Summary_data\\density\\' + study + '\\'
    file = 'All_' + visit + '_' + window_text + '_density_stats.csv'
    final_stats1 = pd.read_csv(path_density + 'All_01_win_10s_step_1s__density_stats.csv')
    final_stats2 = pd.read_csv(path_density + 'All_01_win_10s_step_1s__cropped_density_stats.csv')
    merged = final_stats1.merge(final_stats2, on='subject')
    corr = merged["50%_x"].corr(merged["50%_y"])
    print('n: '+str(len(merged))+ ' r: '+ str(corr))
    plt.scatter(merged["50%_x"], merged["50%_y"])
    plt.xlabel("df1 value")
    plt.ylabel("df2 value")
    plt.show()

if stride_time:

    # calc and plot / file pre_density
    path_density = nimbal_drive + paper_path + 'created_data\\stride_time\\' + study + '\\'
    results = []
    for subj in subj_list:
        print('\tSubject: \t' + subj)
        file = subj + '_' + visit + '_stride_time.csv'
        raw_data = pd.read_csv(path_density + file)

        # flatten to 1 array
        combined = raw_data.stack()
        # drop NA , 0 and values
        combined = raw_data.stack()
        clean = combined[(combined > 0) & (combined < 2)]
        # convert na to zeros
        clean = clean.fillna(0)
        n = len(clean)
        if clean.dropna().empty:
            print(f"No data for {subj}, skipping")
            continue

        # calculate peak based on KDE
        kde = gaussian_kde(clean)
        # Evaluate KDE on a grid
        x = np.linspace(clean.min(), clean.max(), 50)

        y = kde(x)
        #plt.plot(x, y)

        # Normalize density
        y_norm = y / y.sum()

        # Peak = x where density is highest
        peak = x[np.argmax(y_norm)]

        #Weighted mean (≈ KDE mean)
        mean = np.sum(x * y_norm)

        # Weighted variance
        variance = np.sum((x - mean)**2 * y_norm)

        # Append row
        results.append({"subject": subj, "mode": peak, "mean": mean, "variance": variance, "sample_size": n})

    results = pd.DataFrame(results)
    results["std"] = results["variance"] ** 0.5

    #will plot the KDE if want to vie it whiel processing - but slows thinsg down alot - good for debugging
    if plot_stride_time:
        fig = plt.figure(figsize=(12, 8))

        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
        # Left subplot (raster plot)
        ax1 = fig.add_subplot(gs[0, 0])
        # Right subplot (sample size bar plot)
        ax2 = fig.add_subplot(gs[0, 1])

        #plt.figure(figsize=(10, 8))
        #sns.set(style="whitegrid")
        # Sort subjects so the raster is clean

        results = results.sort_values("mode")
        ax1.errorbar(x=results["mode"], y=results["subject"], xerr=results["std"], fmt="o", ecolor="gray", capsize=3, markersize=3, color="steelblue" )
        ax1.set_xlabel("Stride time (secs)")
        ax1.set_ylabel("")
        ax1.set_yticks([])

        # Right: sample size bars
        y_positions = range(len(results))
        ax2.barh(y_positions, results["sample_size"], color="lightgray", edgecolor="black")
        # Only show min and max x‑axis labels
        xmin = results["sample_size"].min()
        xmax = results["sample_size"].max()
        ax2.set_xticks([xmin, xmax])

        ax2.set_xlabel("# strides")
        ax2.set_yticks([]) # no subject labels ax2.set_title("Sample Size per Subject")
        ax2.set_ylabel("")

        #plt.title("Subject Raster: Mode with Horizontal Variance Bars")
        plt.tight_layout()
        plt.show()

    #density data to determin time in each
    # catgory: none - 0
    #          step >0 to density < peak - std
    #          walk > peak - std
    path_density = nimbal_drive + paper_path + 'created_data\\density\\' + study + '\\'
    output = []
    intensity_steps = []
    intensity_walks = []
    intensity_all = []

    for subj in subj_list:
        print('\tSubject: \t' + subj)

        #find values from results panda that has stride time details
        stride_time_row = results[results['subject'] == subj]
        cut_point = stride_time_row['mode'].iloc[0] - (1.96*(stride_time_row['variance'].iloc[0]**0.5))

        file = subj + '_' + visit + '_' + window_text + '_density.csv'
        raw_data = pd.read_csv(path_density + file)
        raw_data = raw_data.iloc[2:].reset_index(drop=True)
        raw_data = raw_data.iloc[:, 1:]
        # convert to density  strides / sec

        # flatten to 1 array
        #combined = density_subj.stack()
        # convert na to zeros
        raw_data = raw_data.fillna(0)
        density = (1 / raw_data).where(raw_data != 0, 0)
        #density = window_size / raw_data
        if density.dropna().empty:
            print(f"No data for {subj}, skipping")
            continue

        total_n = density.size
        no_steps = (density == 0).sum().sum()
        steps_mask = ((density > 0) & (density < cut_point))
        steps = steps_mask.sum().sum()
        steps_density = density[steps_mask]

        walks_mask = (density > cut_point)
        walks = walks_mask.sum().sum()
        walks_density = density[walks_mask]

        rotated = walks_density.T
        intensity_walks.append(rotated)
        rotated = steps_density.T
        intensity_steps.append(rotated)
        rotated = density.T
        intensity_all.append(rotated)

        # Append row
        output.append({"subject": subj,"mode": stride_time_row['mode'].iloc[0],
                      "95CI": 1.96*stride_time_row['variance'].iloc[0]**0.5,
                       "cut_point": cut_point, "total_n": total_n, "No_steps": no_steps,
                       "low_steps": steps, "high_walks": walks})
    output = pd.DataFrame(output)
    output.to_csv(path+"step_classification_pref_v1.csv", index=False)


    if plot_density2:


        plt.figure(figsize=(10, 6))
        intensity_matrix = pd.concat(intensity_all, axis=0, ignore_index=True)
        intensity_matrix = intensity_matrix.apply(pd.to_numeric, errors='coerce')
        intensity_matrix = intensity_matrix.fillna(0)
        #intensity_array = intensity_matrix.to_numpy(dtype=float)
        plt.imshow(intensity_matrix, aspect='auto', cmap='viridis')  # or 'hot', 'plasma', 'magma'
        plt.title("All steps")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        intensity_matrix = pd.concat(intensity_steps, axis=0, ignore_index=True)
        intensity_matrix = intensity_matrix.apply(pd.to_numeric, errors='coerce')
        intensity_matrix = intensity_matrix.fillna(0)
        # intensity_array = intensity_matrix.to_numpy(dtype=float)
        plt.imshow(intensity_matrix, aspect='auto', cmap='viridis')  # or 'hot', 'plasma', 'magma'
        plt.title("Steps only")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        intensity_matrix = pd.concat(intensity_steps, axis=0, ignore_index=True)
        intensity_matrix = intensity_matrix.apply(pd.to_numeric, errors='coerce')
        intensity_matrix = intensity_matrix.fillna(0)
        # intensity_array = intensity_matrix.to_numpy(dtype=float)
        plt.imshow(intensity_matrix, aspect='auto', cmap='viridis')  # or 'hot', 'plasma', 'magma'
        plt.title("Walks only")
        plt.tight_layout()
        plt.show()
