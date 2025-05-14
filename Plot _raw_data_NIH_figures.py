import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from Functions import wake_sleep, bout_bins, steps_by_day, step_density_sec,read_demo_ondri_data, read_demo_data
import numpy as np
import seaborn as sns
import datetime
import openpyxl


#set up paths
root = 'W:'
#check - but use this one - \prd\nimbalwear\OND09
#path1 = root+'\\prd\\NiMBaLWEAR\\OND09\\analytics\\'
path1 = root+'\\prd\\NiMBaLWEAR021\\SA-PR01\\analytics\\'

study = 'SA-PR01'

subj = '10001'
visit= 'week4'
sensor = 'AXV6'
nhours = 10

raw_path_drive = 'W:\\prd\\nimbalwear021\\SA-PR01\\wearables\\device_edf_cropped\\'

