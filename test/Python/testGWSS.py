import csv
import sys
from pygwmodel.gwss import GWSS
from pygwmodel.mat_interface import numpy2mat

londonhp_path = sys.argv[1]
with open(londonhp_path) as londonhp_file:
    reader = csv.DictReader(londonhp_file)
    for item in reader:
        pass
