interval_length = 400  # The average distance between each R-wave. Also the size of the input given into the model
stack = 8  # How many signals to stack, to allow the model to use information from previous signals to predict for
# the current signal
step = 200  # The amount of data to jump by (before scale_down). For example, (0, 400) -> (200, 600)
scale_down = 4  # Optional, to reduce the amount of data. Averages every (scale_down) datapoints into one.
datapoints = interval_length // scale_down  #
lines_per_file = 5000000  # The number of lines per file the model_prediction program creates
max_dist_percentage = 0.2  # Maximum amount the R peaks can vary
T = 0.1  # Remaining are the filter parameters, taken from
# https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units and
# experimented with to find the current parameters.
fs = 4000.0
nyq = fs * 0.5
high_cutoff = 5
low_cutoff = int(nyq / high_cutoff)
order = 4
