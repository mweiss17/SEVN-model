import csv
import argparse
from comet_ml import Experiment
parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--comet', default='', help='add comet.ml credentials in the format workspace/project/api_key')
args = parser.parse_args()
comet_credentials = args.comet.split("/")
experiment = Experiment(
    api_key=comet_credentials[2],
    project_name=comet_credentials[1],
    workspace=comet_credentials[0])

with open('results.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if len(row) == 2:
            experiment.log_parameter(row[0], row[1])
        else:
            experiment.log_metric(row[0], row[1], step=row[2])
