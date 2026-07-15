import pandas as pd

study = 'OND09'
sub_set = 'pd_subset'
root = 'W:'

# sensor data path location
input_path = root + '\\nimbalwear\\' + study + '\\wearables\\device_edf_cropped\\'  # beth_posture_test\\'
# Gait and analytics folders
gait_path = root + '\\nimbalwear\\' + study + '\\analytics\\gait\\steps\\'

# out put target folder - bouts for the combined and site for the individul sensor locations
bouts_output_path = root + '\\Annotated posture\\' + study + '\\' + sub_set + '\\analytics\\posture\\bouts\\'
sites_output_path = root + '\\Annotated posture\\' + study + '\\' + sub_set + '\\analytics\\posture\\sites\\'

subject_id = 'SBH0279'
steps_file = gait_path + study+"_"+subject_id+"_01_GAIT_STEPS.csv"
posture_bouts_file = bouts_output_path + study +"_"+subject_id+"_01_posture_combined_bouts.csv"

# Load files
bouts = pd.read_csv(posture_bouts_file)
steps = pd.read_csv(steps_file)

steps_unbouted = steps[steps["gait_bout_num"] == 0]
steps_bouted = steps[steps["gait_bout_num"] > 1]

print("Total -  Unbouted:" + str(len(steps_unbouted)) + "   Bouted: " + str(len(steps_bouted)))

for name, steps in [("bouted", steps_bouted), ("unbouted", steps_unbouted)]:

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

    print(name)
    print(steps_by_posture)
