import pandas as pd
from Functions import clustering, get_demo_characteristics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

###############################################################
study = 'SA-PR01'
nimbal_dr = 'O:'
sub_study = 'AAIC 2025'
paper_path = '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'
summary_path = nimbal_dr + paper_path + 'Summary_data\\'

##############################################################
#read in subject list from file in summary drive
subjects = pd.read_csv(summary_path+'subject_ids_'+sub_study+'.csv')

###########################################
# read in step bout data
summary_steps_file = study+'_bout_steps_daily_bins_with_unbouted.csv'
steps_data = pd.read_csv(summary_path+summary_steps_file)
steps_data = steps_data.rename(columns={'subj': 'SUBJECT'})

summary_dur_file = study +'_bout_width_daily_bins_with_unbouted.csv'
width_data = pd.read_csv(summary_path+summary_dur_file)
width_data = width_data.rename(columns={'subj': 'SUBJECT'})

##############################################
# Run the analysis on bout steps or bout width
by_step = False

if by_step:
    steps_all = steps_data[steps_data['all/sleep']=='all']
    graph_title1 = 'Bouts by stride count'
    file_out =  'Bouts_by_step_summary_by_subject_7days.csv'
else:
    #using width in time
    steps_all = width_data[width_data['all/sleep']=='all']
    graph_title1 = 'Bouts by seconds'
    file_out = 'Bouts_by_sec_summary_by_subject_7days.csv'

counts = steps_all['SUBJECT'].value_counts().reset_index()
counts = counts.reset_index()


################################################################################
# this sets upf fiels for plott ing and analysis
#select all steps that match the subject list
step1 = steps_all[steps_all['SUBJECT'].isin(subjects['SUBJECT'])]
# select only those with first 7 days
#step1 = step1.groupby('SUBJECT').head(7).reset_index(drop=True)

#slect the columns names to plot/analyze - only stride counts fo this
#this lien was for numebr of bouts
#select1 = step1.columns[step1.columns.str.startswith('n_')].tolist()

#these are the strides per bout class
select2 = step1.columns[step1.columns.str.startswith('strides_')].tolist()
#this adds unbouted
select2.insert(0, 'not_bouted')
select2.insert(0, 'total')

#calcualte the percentrage of steps in bouts relatiev to total (daily)
for col in select2:
    step1[col + '_pct'] = step1[col] / step1['total'] * 100

#cols = ['vlow_corr','low_corr','med_corr','high_corr']
short_bouts = step1[['strides_<_5', 'strides_<_10', 'strides_<_30']]
step1['short'] = short_bouts.sum(axis=1)
med_bouts = step1[['strides_<_60', 'strides_<_180']]
step1['medium'] = med_bouts.sum(axis=1)
long_bouts = step1[['strides_<_600', 'strides_>_600']]
step1['long'] = long_bouts.sum(axis=1)

cluster_cols = ['not_bouted', 'short', 'medium', 'long']
subset = step1[['SUBJECT'] + cluster_cols]
subset = subset.groupby(subset['SUBJECT']).sum()

#sum across subject days
cluster_data = subset
ncluster = 3
data_out, labels = clustering(cluster_data, ncluster=ncluster)
x = np.arange(len(cluster_cols))
plt.figure(figsize=(10, 6))
sns.lineplot(data=data_out, x='feature', y='value', hue='cluster', palette='Set2')
plt.xlabel('Density (strides/min)')
plt.ylabel('Proportion of wear time')
#plt.xticks(ticks=x, labels=[cluster_cols])
plt.title('Step density pattern clusters - ALL')
plt.show()

subject_clusters = pd.DataFrame({'Subject': cluster_data.index,'Cluster': labels})

cluster_tables = []
for i in range(ncluster):
    cluster_id = subject_clusters[subject_clusters['Cluster'] == i]
    subjects = cluster_id['Subject']
    print ('Cluster '+str(i) + ' - n: '+ str(len(subjects)))
    summary_table = get_demo_characteristics(study='SA-PR01', sub_study='AAIC 2025', subjects=subjects, group_col=None)
    cluster_tables.append(summary_table)


print ('test')

#####################################################
#MEANS GRAPH
# subject means
#bouts - calculate means for bout #s
#nbout_subj_means = step2.groupby('SUBJECT')[select1].mean()
#nbout_all_means = nbout_subj_means.mean()
#nbout_all_std = nbout_subj_means.std()

#bouts setp #s absolute
nstride_subj_means = step2.groupby('SUBJECT')[select2].mean()
nstride_all_means = nstride_subj_means.mean()
nstride_all_std = nstride_subj_means.std()

#bouts setp #s percentage
nstride_pct_subj_means = step2.groupby('SUBJECT')[select3].mean()
nstride_pct_all_means = nstride_pct_subj_means.mean()
nstride_pct_all_std = nstride_pct_subj_means.std()

#coraltion matrix across bouts strides
#corr_matrix = corr_matrix_all_columns(nstride_subj_means, para=False)
#corr_matrix.to_csv(summary_path+"corr_matrix_bout_strides.csv")

'''
##################################################################
#scatter graphs

fig, axs = plt.subplots(1, 3, figsize=(10, 6))
lab_list = ['not_bouted', 'strides_>_600']
for i in lab_list:
    axs[0].scatter(nstride_subj_means['total'], nstride_subj_means[i], label=i)

lab_list = ['not_bouted_pct', 'strides_>_600_pct']
for i in lab_list:
    axs[1].scatter(nstride_subj_means['total'], nstride_pct_subj_means[i], label=i)

axs[0].set_xlabel('Total strides')
axs[0].set_ylabel('Strides')
axs[0].set_title('Strides per bout vs total strides')
axs[0].legend(loc='right')

axs[1].set_xlabel('Total strides')
axs[1].set_ylabel('% of total strides')
axs[1].set_title('% bouted strides of total')
axs[1].legend(loc='right')

# Plot stacked bars manually
bottom = [0] * len(nstride_subj_means)
x = range(len(nstride_subj_means))
# Sort rows by total row sum (ascending so largest on right)
data_sorted = nstride_subj_means.copy()
data_sorted['Total'] = nstride_subj_means.sum(axis=1)
data_sorted = data_sorted.sort_values('Total')
data_sorted = data_sorted.drop(columns='Total')

temp = data_sorted.iloc[:, 1:]
for col in temp.columns:
    axs[2].bar(x, temp[col], bottom=bottom, label=col)
    bottom = [i + j for i, j in zip(bottom, temp[col])]

plt.tight_layout()
plt.show()
print('pause')



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


print('pause')

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
plt.show()'''

