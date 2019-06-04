import os
import csv

def save_metric(row, filename):
    create_stat_dir_if_not_exists()
    filename = 'stats/' + filename + '.csv'

    with open(filename, 'a') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(row)

    writeFile.close()


def create_stat_dir_if_not_exists():
    directory = 'stats'
    if not os.path.exists(directory):
        os.makedirs(directory)