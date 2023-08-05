from configparser import SafeConfigParser
import configparser
import os

parser = SafeConfigParser()
parser.read('selector.ini')
animal = parser.get('Animal', 'animal')

parser.read(f'{animal}_config.ini')

interval_length = int(parser.get('Signal Parameters', 'interval_length'))

stack = int(parser.get('Model Parameters', 'stack'))
scale_down = int(parser.get('Model Parameters', 'scale_down'))
window_size = int(parser.get('Model Parameters', 'window_size'))
datapoints = window_size // scale_down

lines_per_file = int(parser.get('Model Prediction Parameters', 'lines_per_file'))
max_dist_percentage = float(parser.get('Model Prediction Parameters', 'max_dist_percentage'))
threshold = float(parser.get('Model Prediction Parameters', 'threshold'))

T = float(parser.get('Filter Parameters', 'T'))
fs = int(parser.get('Filter Parameters', 'fs'))
nyq = fs * 0.5
high_cutoff = int(parser.get('Filter Parameters', 'high_cutoff'))
low_cutoff = int(parser.get('Filter Parameters', 'low_cutoff'))
order = int(parser.get('Filter Parameters', 'order'))

if __name__ == '__main__':
    # To change the selector
    cwd = os.getcwd()
    for file in os.listdir(cwd):
        if file.endswith('.ini') and not file.startswith('selector'):
            print(file)
    print('Existing configs (change the selector to match one of the prefixes)')
    animal = input()
    if os.path.exists(f'{animal}_config.ini'):
        config = configparser.ConfigParser(allow_no_value=True)
        config.read('selector.ini')
        config.set('Animal', '; Should be the prefix of the config file the program should use, as in {'
                             'animal}_config.ini is what is loaded by the program')
        config.set('Animal', 'animal', animal)
        with open('selector.ini', 'w') as configfile:
            config.write(configfile)
