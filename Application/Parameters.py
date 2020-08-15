import re


file = open('parameters.txt', 'r')

interval_length = int(re.findall('([-0-9.]+)', file.readline())[0])
max_dist_percentage = float(re.findall('([-0-9.]+)', file.readline())[0])
lines_per_file = int(re.findall('([-0-9.]+)', file.readline())[0])