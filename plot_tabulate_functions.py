import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Sample summary table
def table_plot(summary_table):
    # Set figure size
    plt.figure(figsize=(8, 6))

    # Bar positions
    x = np.arange(len(summary_table.index))

    # Plot bars for mean values
    plt.bar(x - 0.2, summary_table['mean_X'], yerr=summary_table['std_X'], width=0.4, capsize=5, label='X', color='b', alpha=0.7)
    plt.bar(x + 0.2, summary_table['mean_Y'], yerr=summary_table['std_Y'], width=0.4, capsize=5, label='Y', color='r', alpha=0.7)

    # Labels and title
    plt.xlabel("Group")
    plt.ylabel("Values")
    plt.title("Histogram of Mean Values with Standard Deviation Bars")
    plt.xticks(ticks=x, labels=summary_table.index)  # Set x-axis labels
    plt.legend()

    # Show plot
    plt.show()

def plot_density_raw (summary_path, files, bouts_all, demodata):
    # bout density data
    # plot density summary file
    data_path = summary_path+'density\\'

    count = 0
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(10,6))
    time = np.linspace(1, 1440, 1440)

    for index, file in enumerate(files):
        #print(f'\rSubj #: {index}' + ' of ' + str(len(files)), end='', flush=True)
        parts = file.split('_')
        subj = parts[0] + parts[1]

        #cohort = demodata.loc[demodata['SUBJECT'] == subj, 'COHORT'].values[0]
        #age = demodata.loc[demodata['SUBJECT'] == subj, 'AGE'].values[0]

        #sub_set = bouts_all[bouts_all['subj'] == subj]
        #n_days = len(sub_set)
        #tot_steps = sub_set['total'].sum()
        #tot_steps_day = tot_steps / n_days

        print(f'\rSubj #: {subj} - {index} of ' + str(len(files)), end='', flush=True)
        subj_density = pd.read_csv(data_path+file)
        del subj_density[subj_density.columns[0]]
        subj_density = subj_density.loc[1:,:]
        # plot by day and subject
        for day, col in enumerate(subj_density.columns):
            data1 = subj_density[col].values
            data1 = data1.astype(int)
            data2 = data1.reshape(1, -1)
            ax.imshow(data2, aspect = 'auto', cmap='viridis', interpolation=None, extent=[time[0], time[-1], count, count+1])
            #print(f'\rCount #: {count}', end='', flush=True)
            count=count+1
        count=count+1
    ax.set_ylim(0, count)
    ax.set_ylabel("Subjects - and days within subjects")
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Time (mins/day)")
    #ax.set_title("Step Density time series - Subject: "+ subj+"  Cohort: "+cohort+"  Age: "+str(age))
    ax.set_title("Step Density time series - ALL  > 10,000 steps/day")
    plt.colorbar(ax.images[0], ax=ax, label='Density (strides/minute)')  # colorbar from the first image


    plt.tight_layout()
    plt.show()
    #plt.savefig(data_path+'images\\'+subj+'_density_days.pdf')
print()






'''

#these were for ANG

fig, ax = plt.subplots(1,4)
fig.tight_layout(pad=2)

ax[0].set_ylim([0, 10.5])
ax[0].tick_params(bottom=False)
ax[0].set_title('Sleep duration', fontsize = 11)
ax[0].set_ylabel('Hours / day')
ax[0].set_xlabel('')
sleep = daily['Sdur_avg']/60
sns.boxplot(sleep,color='greenyellow', ax=ax[0],width=0.6)

ax[1].set_ylim([0, 22000])
ax[1].set_title('Steps', fontsize = 11)
ax[1].tick_params(bottom=False)
ax[1].set_ylabel('Number / day')
sns.boxplot(daily['nstep_avg'],color='c', ax=ax[1],width=0.6)

ax[2].set_ylim([0, 155])
ax[2].set_title('Activity -MVPA', fontsize = 11)
ax[2].tick_params(bottom=False)
ax[2].set_ylabel('Minutes / day')
sns.boxplot(daily['Mod_avg'],color='orange', ax=ax[2],width=0.6)

ax[3].set_ylim([0, 16])
ax[3].set_title('Sedentary', fontsize = 11)
ax[3].tick_params(bottom=False)
ax[3].set_ylabel('Hours / day')
sedentary = daily['Sed_avg']/60
sns.boxplot(sedentary,color='plum', ax=ax[3],width=0.6)
plt.show()


#these for karen
pal = sns.color_palette(n_colors=2)
SA_Control = daily[daily["Group"] != 'zNA']
h = sns.jointplot(data=SA_Control, x="nstep_avg", y="Sed_avg", hue="Group", s=50,
              xlim=(-1000, 13000), ylim=(400, 1000), palette=pal)
h.set_axis_labels('Steps / day', 'Sedentary time - mins / day', fontsize=12)
for Group, color in zip(['Superager','Control'], pal):
   sns.regplot(data=daily[daily['Group'] == Group], x="nstep_avg", y="Sed_avg", color=color, truncate = False, ax=h.ax_joint, ci=None)

plt.show()

h = sns.jointplot(data=SA_Control, x="nstep_med", y="Sed_med", hue="Group", s=50,
              xlim=(-1000, 13000), ylim=(400, 1000))
h.set_axis_labels('Steps / day', 'Sedentary time - mins / day', fontsize=12)
plt.show()

h = sns.jointplot(data=SA_Control, x="nstep_avg", y="Sdur_avg", hue="Group", s=50,
              xlim=(-1000, 13000), ylim=(200, 700))
h.set_axis_labels('Steps / day', 'Sleep time - mins / night', fontsize=12)
plt.show()

h = sns.jointplot(data=SA_Control, x="nstep_med", y="Sdur_med", hue="Group", s=50,
              xlim=(-1000, 13000), ylim=(200, 700))
h.set_axis_labels('Steps / day', 'Sleep time - mins / night', fontsize=12)
plt.show()

h = sns.jointplot(data=SA_Control, x="Sed_avg", y="Sdur_avg", hue="Group", s=50,
              xlim=(400, 1000), ylim=(200, 700))
h.set_axis_labels('Sedentary - mins / day', 'Sleep time - mins / night', fontsize=12)
plt.show()

h = sns.jointplot(data=SA_Control, x="Sed_med", y="Sdur_med", hue="Group", s=50,
              xlim=(400, 1000), ylim=(200, 700))
h.set_axis_labels('Sedentary - mins / day', 'Sleep time - mins / night', fontsize=12)
plt.show()

h = sns.jointplot(data=SA_Control, x="nstep_avg", y="nstep_sd", hue="Group", s=50,
              xlim=(-999, 13000), ylim=(-999, 8000),kind="reg" )
h.set_axis_labels('Steps / day', 'SD Steps / day', fontsize=12)
plt.show()


pal = sns.color_palette(n_colors=2)
SA_Control = daily[daily["Group"] != 'zNA']

h = sns.jointplot(data=SA_Control, x="nstep_med", y="nstep_sd", hue="Group", s=50,
              xlim=(-999, 13000), ylim=(-999, 8000))
for Group, color in zip(['Control','Superager'], pal):
   sns.regplot(data=daily[daily['Group'] == Group], x="nstep_med", y="nstep_sd", color=color, truncate = False, ax=h.ax_joint, ci=None)
h.set_axis_labels('Steps / day', 'SD Steps / day', fontsize=12)
plt.show()


h = sns.jointplot(data=SA_Control, x="Mod_med", y="Mod_sd", hue="Group", s=50,
              xlim=(-19, 110), ylim=(-19, 80))
for Group, color in zip(['Control','Superager'], pal):
   sns.regplot(data=daily[daily['Group'] == Group], x="Mod_med", y="Mod_sd", color=color, truncate = False, ax=h.ax_joint, ci=None)
h.set_axis_labels('MVPA mins / day', 'SD MVPA mins / day', fontsize=12)
plt.show()



h = sns.jointplot(data=SA_Control, x="Sdur_med", y="Sdur_sd", hue="Group", s=50,
              xlim=(250, 750), ylim=(0, 200))
for Group, color in zip(['Control','Superager'], pal):
   sns.regplot(data=daily[daily['Group'] == Group], x="Sdur_med", y="Sdur_sd", color=color, truncate = False, ax=h.ax_joint, ci=None)
h.set_axis_labels('Sleep mins / day', 'SD Sleep mins / day', fontsize=12)
plt.show()







h = sns.jointplot(data=SA_Control, x="Mod_avg", y="Mod_sd", hue="Group", s=50,
              xlim=(-19, 110), ylim=(-19, 80))
h.set_axis_labels('MVPA mins / day', 'SD MVPA mins / day', fontsize=12)
plt.show()



h = sns.jointplot(data=SA_Control, x="Sdur_avg", y="Sdur_sd", hue="Group", s=50,
              xlim=(250, 750), ylim=(0, 200))
h.set_axis_labels('Sleep mins / day', 'SD Sleep mins / day', fontsize=12)
plt.show()




sns.jointplot(data=daily, x="nstep_med", y="nstep_sd")


sns.jointplot(data=daily, x="Sed_med", y="Sed_sd", hue="Class")

plt.errorbar(SA_Control["nstep_avg"], SA_Control["Sdur_avg"],xerr=SA_Control["nstep_sd"], yerr=SA_Control["Sdur_sd"], fmt="o")
plt.show()


# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(daily['nstep_avg'],daily['Sed_avg'],daily['Mod_avg'])
ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])
plt.show()


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# plot violin plot
axs[0].violinplot(daily['nstep_avg'], showmeans=False, showmedians=True)
axs[0].set_title('Daily steps - Average')
axs[1].violinplot(daily['nstep_med'], showmeans=False, showmedians=True)
axs[1].set_title('Daily steps - median')


# adding horizontal grid lines
#for ax in axs:
#    ax.yaxis.grid(True)
#    ax.set_xticks([y + 1 for y in range(len(all_data))],
#3                  labels=['x1', 'x2', 'x3', 'x4'])
#    ax.set_xlabel('Four separate samples')
#    ax.set_ylabel('Observed values')

plt.show()

'''