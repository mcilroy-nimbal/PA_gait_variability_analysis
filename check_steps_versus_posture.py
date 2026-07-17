import pandas as pd
from pathlib import Path
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

final_df = pd.read_csv("W:\\Annotated posture\\testing_unbouted_posture\\processed_0716.csv")
g = sns.catplot(data=final_df, x="step_count", y="posture_group", hue="bouted", kind="boxen")
g.set_axis_labels("Step_count", "Posture")
g.set_titles("Bouted = {col_name}")
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


'''


study = 'OND09'
#sub_set = 'pd_subset'
sub_set = "all"
root = 'W:'

# sensor data path location
input_path = root + '\\nimbalwear\\' + study + '\\wearables\\device_edf_cropped\\'  # beth_posture_test\\'
# Gait and analytics folders
gait_path = root + '\\nimbalwear\\' + study + '\\analytics\\gait\\steps\\'

# out put target folder - bouts for the combined and site for the individul sensor locations
bouts_output_path = root + '\\Annotated posture\\' + study + '\\' + sub_set + '\\analytics\\posture\\bouts\\'
sites_output_path = root + '\\Annotated posture\\' + study + '\\' + sub_set + '\\analytics\\posture\\sites\\'
output_path = Path(root + '\\Annotated posture\\testing_unbouted_posture\\processed_0716.csv')

# if doing 1 subject
#subject_id = 'SBH0279'

#doing all that have posture bout data
root = Path(bouts_output_path)
bout_files = [p.name for p in root.rglob("*.csv")]

subjects = []
for p in bout_files:
    stem = os.path.splitext(os.path.basename(p))[0]
    parts = stem.split("_")
    subjects.append(parts[1])
unique_subjects = list(set(subjects))
print('Number of subjects found: ' + str(len(unique_subjects)))

df_list = []
for subj in unique_subjects:

    steps_file = gait_path + study+"_"+subj+"_01_GAIT_STEPS.csv"
    posture_bouts_file = bouts_output_path + study +"_"+subj+"_01_posture_combined_bouts.csv"

    # Load files
    bouts = pd.read_csv(posture_bouts_file)
    steps = pd.read_csv(steps_file)

    steps_unbouted = steps[steps["gait_bout_num"] == 0]
    steps_bouted = steps[steps["gait_bout_num"] > 1]

    print("Subject: "+subj+"  Total steps: -  Unbouted:" + str(len(steps_unbouted)) + "   Bouted: " + str(len(steps_bouted)))



    for name, steps in [("bouted", steps_bouted), ("unbouted", steps_unbouted), ("all", steps)]:

        # Convert to datetime
        bouts["timestamp"] = pd.to_datetime(bouts["timestamp"])
        bouts["end_timestamp"] = pd.to_datetime(bouts["end_timestamp"])
        steps["step_time"] = pd.to_datetime(steps["step_time"])

        mapping = {"stand":"stand", "transition":"stand", "sit":"sit", "sitstand":"sitstand", "reclined":"reclined",
            "supine":"lying", "leftside":"lying", "rightside":"lying", "prone":"lying"}
        bouts["posture_group"] = bouts["posture"].map(mapping)

        # Sort step times
        step_times = steps["step_time"].sort_values()

        # Count steps within each bout
        bouts["step_count"] = bouts.apply(lambda row: ((step_times >= row["timestamp"]) & (step_times <= row["end_timestamp"])).sum(), axis=1)

        steps_by_posture = (bouts.groupby("posture_group")[["step_count", "duration"]].sum().reset_index())

        total_steps = steps_by_posture["step_count"].sum()
        total_duration = steps_by_posture["duration"].sum()

        steps_by_posture["percent_steps"] = (100 * steps_by_posture["step_count"] / total_steps).round(1)
        steps_by_posture["percent_time"] = (100 * steps_by_posture["duration"] / total_duration).round(1)
        steps_by_posture["subj"] = subj
        steps_by_posture['bouted'] = name

        print(name)
        print(steps_by_posture)

        df_list.append(steps_by_posture)

# Combine all dataframes into one
final_df = pd.concat(df_list, ignore_index=True)

output_path = Path(root + '\\Annotated posture\\testing_unbouted_posture\\processed_0716.csv')
final_df.to_csv(output_path, index=False)

final_df.drop(columns=['subj'], inplace=True)
summary = final_df.groupby(['bouted', 'posture_group']).agg(['mean', 'median', 'std', 'count'])
print(summary)
'''
