import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from Functions import read_demo_ondri_data
from scipy.stats import gaussian_kde
import seaborn as sns
import datetime

###############################################################
study = 'OND09'
nimbal_dr = 'O:'

###########################################
#read in the cleaned data file for the HANNDS methods paper
demo_path = nimbal_dr+'\\Papers_NEW_April9\\Shared_Common_data\\'+study+'\\'
#reads this file - 'OND09_ALL_01_CLIN_DEMOG_2025_CLEAN_HANDDS_METHODS_N245.csv'
demodata = read_demo_ondri_data(demo_path)

def remove_underscore(s):
    parts = s.split('_')
    fixed = parts[0] +'_'+ parts[1] + parts[2]
    return fixed
demodata['SUBJECT'] = demodata['SUBJECT'].apply(remove_underscore)

###########################################
# read in step bout data
paper_path = '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'
summary_path = nimbal_dr + paper_path + 'Summary_data\\'

summary_steps_file = study+'_bout_steps_daily_bins_with_unbouted.csv'
steps_data = pd.read_csv(summary_path+summary_steps_file)
steps_data = steps_data.rename(columns={'subj': 'SUBJECT'})

summary_dur_file = study +'_bout_width_daily_bins_with_unbouted.csv'
width_data = pd.read_csv(summary_path+summary_dur_file)
width_data = width_data.rename(columns={'subj': 'SUBJECT'})

##############################################
# select subset of main data based on
# Count the number of rows for each unique SubjectCode
by_step=False

if by_step:
    steps_all = steps_data[steps_data['all/sleep']=='all']
    graph_title1 = 'Bouts by stride count'
else:
    #using width in time
    steps_all = width_data[width_data['all/sleep']=='all']
    graph_title1 = 'Bouts by seconds'

counts = steps_all['SUBJECT'].value_counts().reset_index()
counts = counts.reset_index()

#table of the counts
merged = pd.merge(counts, demodata, on='SUBJECT')
merged = merged[merged['COHORT'] == 'Community Dwelling']

# Define age bins and labels
bins = [0, 50, 60, 70, 80, 100]
labels = ['<50', '50-60', '60-70', '70-80','>80']

# Create new column for age groups
merged['AgeGroup'] = pd.cut(merged['AGE'], bins=bins, labels=labels, right=False)

merged = merged[(merged['AGE'] > 49) & (merged['count'] > 6)]
print(len(merged))

# Summary calculations
sex_counts = merged['SEX'].value_counts(normalize=True).mul(100).round(1)
avg_age = merged['AGE'].mean()
std_age = merged['AGE'].std()
marital_counts = merged['MRTL_STATUS'].value_counts(normalize=True).mul(100).round(1)

# Get unique values in 'Category'
a = merged['MRTL_STATUS'].unique()
mrtl_list = a.tolist()

work_counts = merged['EMPLOY_STATUS'].value_counts(normalize=True).mul(100).round(1)
a = merged['EMPLOY_STATUS'].unique()
employ_list = a.tolist()

# Build summary table
summary = pd.DataFrame({
    'Metric': ['Avg age', 'Std_age', '% Male', '% Female'] + [f'Count in {label}' for label in mrtl_list]+ [f'Count in {label2}' for label2 in employ_list],
    'Value': [
        round(avg_age, 2),
        round(std_age, 2),
        sex_counts.get('Male', 0),
        sex_counts.get('Female', 0),
        *[marital_counts.get(label, 0) for label in mrtl_list],
        *[work_counts.get(label, 0) for label in employ_list]
    ]
})

# Create a blank row
blank_row = pd.DataFrame({'Metric': [''], 'Value': ['']})
summary = pd.concat([summary.iloc[:4], blank_row, summary.iloc[4:]]).reset_index(drop=True)
summary = pd.concat([summary.iloc[:11], blank_row, summary.iloc[11:]]).reset_index(drop=True)
print (summary)

target_subj = merged['SUBJECT'].unique()
subjects = pd.Series(target_subj)

################################################################################
# stride time plots and analysis
plot_rows = []
x_vals = np.linspace(0, 5, 250)
plt.figure(figsize=(10, 6))

for i in subjects:
    full=[]
    print ('Subject: ' + str(i))
    stride_time = pd.read_csv(summary_path + 'stride_time\\' + str(i) + '_01_stride_time.csv')
    stride_time = stride_time.drop(columns=['Unnamed: 0'])
    if stride_time.shape[1] > 7:
        cols_to_drop = stride_time.columns[7:]  # everything after the 7th
        stride_time = stride_time.drop(columns=cols_to_drop)
    for col_name, array in stride_time.items():
        print (col_name)
        array = array[~np.isnan(array)]
        full.extend(array)

        #total = len(array)
        #array = array[array <= 3]
        #less3 = len(array)
        #print (total, less3)
        # Compute histogram per day
        #hist, bin_edges = np.histogram(array, bins=100, range = (0, 3), density=True)
        #plot_rows.append(hist)

    kde = gaussian_kde(full,bw_method=0.001)
    density = kde(x_vals)
    peak_index = np.argmax(density)
    peak_x = x_vals[peak_index]
    peak_y = density[peak_index]


    plt.plot(x_vals, density)
    #hist, bin_edges = np.histogram(full, bins=30, range = (0, 5), density=True)
    #plot_rows.append(hist)
# Stack to 2D array
#heatmap_data = np.vstack(plot_rows)  # shape: (total_rows, bins)

# Plot heatmap
#plt.figure(figsize=(12, 20))
#plt.imshow(heatmap_data, aspect='auto', interpolation='none', cmap='plasma', origin='lower')
#plt.colorbar(label='Frequency Density')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Density plot per subject')
plt.tight_layout()
plt.show()




################################################################################
#MEANS graphs

#all steps
step1 = steps_all[steps_all['SUBJECT'].isin(merged['SUBJECT'])]

#first 7 days
step2 = step1.groupby('SUBJECT').head(7).reset_index(drop=True)


#slect the columns names to plot/analyze
# these are the bout #s
select1 = step2.columns[step2.columns.str.startswith('n_')].tolist()

#these are the strides per bout class
select2 = step2.columns[step2.columns.str.startswith('strides_')].tolist()
#this adds unbouted
select2.insert(0, 'not_bouted')

#calcualte the percentrage of stpes in bouts relatiev to total (daily)
for col in select2:
    step2[col + '_pct'] = step2[col] / step2['total'] * 100

#this is the header for the percentage valaues
select3 = step2.columns[step2.columns.str.endswith('_pct')].tolist()


#####################################################
#MEANS GRAPH
# subject means
#bouts - calculate means for bout #s
nbout_subj_means = step2.groupby('SUBJECT')[select1].mean()
nbout_all_means = nbout_subj_means.mean()
nbout_all_std = nbout_subj_means.std()

#bouts setp #s absolute
nstride_subj_means = step2.groupby('SUBJECT')[select2].mean()
nstride_all_means = nstride_subj_means.mean()
nstride_all_std = nstride_subj_means.std()

#bouts setp #s percentage
nstride_pct_subj_means = step2.groupby('SUBJECT')[select3].mean()
nstride_pct_all_means = nstride_pct_subj_means.mean()
nstride_pct_all_std = nstride_pct_subj_means.std()





##################################################################
#scatter graphs


# plt.scatter(x, unbout, color='red', label = 'unbouted')
plt.scatter(x, short, color='orange', label='short <10')
# plt.scatter(x, med, color='blue', label='med 10-50')
plt.scatter(x, long, color='green', label='long >50')


'''

# Step 2: Create the plot
fig,axs = plt.subplots(3, figsize=(8, 9))

axs[0].bar(nbout_all_means.index, nbout_all_means.values, yerr=nbout_all_std.values, capsize=5, color='lightgreen', edgecolor='black')
axs[0].set_title(graph_title1 + ' - Mean # bouts / day')
axs[0].set_xlabel('Bout labels')
axs[0].set_ylabel('Bouts / day')

axs[1].bar(nstride_all_means.index, nstride_all_means.values, yerr=nstride_all_std.values, capsize=5, color='lightblue', edgecolor='black')
axs[1].set_title(graph_title1 + ' - Mean # strides / day')
axs[1].set_xlabel('Bout labels')
axs[1].set_ylabel('Strides / day')

axs[2].bar(nstride_pct_all_means.index, nstride_pct_all_means.values, yerr=nstride_pct_all_std.values, capsize=5, color='violet', edgecolor='black')
axs[2].set_title(graph_title1 + ' - Mean % strides / day')
axs[2].set_xlabel('Bout labels')
axs[2].set_ylabel('Strides / day')

plt.tight_layout()
plt.show()
'''

'''
######################################################
#Day ti day varaition - coeeficitn of variation

step2_subj_cv = step2.groupby('SUBJECT')[select2].agg(lambda x: x.std() / x.mean())
step2_all_means = step2_subj_cv.mean()
step2_all_std = step2_subj_cv.std()

# Step 2: Create the plot
fig,axs = plt.subplots(3, figsize=(8, 12))

axs[0].bar(step2_all_means.index, step2_all_means.values, yerr=step2_all_std.values, capsize=5, color='lightgreen', edgecolor='black')

axs[0].set_title(graph_title1 + ' Mean COV (across days)')
axs[0].set_xlabel('Bout labels')
axs[0].set_ylabel('Mean COV (across days)')

plt.tight_layout()
plt.show()
'''







print ('pause')
'''
#######################################################
# mean values (across days - all values from bouts)
steps = True

if steps:
    file = summary_steps_file
else:
    file = summary_dur_file

bout_data = pd.read_csv(summary_path+file)

all = bout_data[bout_data['all/sleep'] == 'all']

if steps:
     subset = all[vars_step_tot]
else:
    subset = all[vars_dur_tot]

grouped = subset.groupby('subj').agg(['mean', 'std'])

means = grouped.xs('mean', axis=1, level=1)
stds = grouped.xs('std', axis=1, level=1)
cvs = stds / means

group_means = means.mean()
group_std = means.std()
group_cvs = cvs.mean()
group_cvs_sd = cvs.std()

# Plot
plt.figure(figsize=(8, 6))

data = group_means
data_std = group_std

plt.bar(data.index, data.values, color='blue')
plt.errorbar(data.index,data.values, yerr=[np.zeros_like(data_std.values), data_std.values],
            fmt='none', ecolor='black', capsize=5 )
for i, col in enumerate(means.columns):
    x = np.random.normal(i, 0.05, size=len(means[col]))  # jittered x-values
    plt.scatter(x, means[col], color='orange', alpha=0.7, label='Data Points' if i == 0 else "")


plt.ylabel('Coefficient of variation')
plt.title('Average Coefficient of variation (across days) per bout')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
'''