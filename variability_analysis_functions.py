import powerlaw
import numpy as np
import matplotlib.pyplot as plt

def inter_day_stab():
    '''
    Inter-day stability
    (Volkers and Scherder, 2011) - Ratio of variance of average 24-h pattern around
    overall mean and overall variance, where higher values indicate more stability
    '''


    return

def intra_day_stab():
    '''
    Intra-day stability
    (Volkers and Scherder, 2011) - Ratio of mean squares of the difference between successive hours
    and the mean squares around the grand mean (overall variance) where lower values indicate better rest-activity pattern
    '''


    return

def alpha_gini_index(data, plot=False):
    '''
    Alpha and Gini (pattern)
    (Barry et al 2015; Chastin and Granat 2010) - alpha = power law distribution exponent (distribution of sedentary and ambulatory bouts according to their time;
    larger alpha indicates distribution of ambulatory bouts is derived from a greater proportion of shorter bouts
    - gini index = accumulation of sedentary and walking time (by bout length); high gini index indicates greater contribution of long bouts
    - See Godfrey supplement in Teams channel for calculations




'''

    # Fit the data to a power-law distribution
    fit = powerlaw.Fit(data)
    # Get the estimated alpha (scaling exponent)
    #print("Alpha (scaling exponent):", fit.alpha)
    #print("Xmin (cutoff value):", fit.xmin)

    #gini
    data = np.sort(data)  # Sort values
    n = len(data)
    cumulative = np.cumsum(data)
    gini = (2 * np.sum((np.arange(1, n + 1) * data)) / (n * np.sum(data))) - (n + 1) / n

    if plot:
        # Plot the data and fitted power law
        fig = plt.figure(figsize=(6, 4))
        fit.plot_pdf(color='b', label='Empirical Data')
        fit.power_law.plot_pdf(color='r', linestyle='--', label='Fitted Power Law')
        plt.legend()
        plt.show()

    return gini, fit.alpha, fit.xmin, n



'''
Diurnal rest-activity rhythm(day/night)
*Note: you probably have a better approach for this as I know you’velooked into this in past 
Relative amplitude of 10 most active consecutive hours and the uninterrupted least active 5 hours 
within a 24h cycle where higher values indicate a larger difference between daytime activity and nighttime rest

Diurnal pattern metrics
(Varma and Watts 2017) - extent to which individual’s walking over a time period (e.g., waking hours) 
deviates from a flat, non-varying rhythm - fluctuations relative to average activity (RMSD) (SD of all min-to-min PA intervals during waking hours

'''

'''
DFA (Cavanaugh et al 2010) - composed of 1-min step count values - 
positive values between 0.5 and 1 = positive persistence in temporal structure 
(data points are positively correlated with one another across multiple time scales) 
based on horizontal structure of each daily time series
'''

'''
ApEN (Cavanaugh et al 2010) - composed of 1-min step count values - quantifies amount of randomness in the 
vertical structure of each daily time series. - Determined probability that short sequences of consecutive 
1-min step counts repeated, at least approximately, throughout the longer temporal 
sequence of 1,440 daily 1-min intervals. - Generates unitless number between 0-2. 
Zero means short sequences are perfectly repeatable (e.g. sine wave). Values of two correspond to time 
series for which repeating sequences of points occur by chance alone.

'''

'''
ER
(Cavanaguh et al 2010) - composed of 1-min step count values - quantifies average amount of uncertainty associated with whether 
any amount of step activity occurred at any given minute - Greater uncertainty implied 
ordering of active versus inactive minutes contained a greater amount of information

'''

'''
CoV

(Cavanaugh et al 2010) - composed of 1-min step counts used to compare against complexity metrics - 
for each daily time series, (CV=100[SD/M]) - independent of order in which step counts were accumulated.

'''


'''

Heat map for bout length

(Paraschiv-Ionescu et al 2018) - as discussed; based on type, intensity, and duration of walking - 
quantified with LZC (Lempel-Ziv computation) and PLZC (Permutation Lempel-Ziv computation) metrics – see Paraschiv-Ionescu supplement in Teams channel

'''