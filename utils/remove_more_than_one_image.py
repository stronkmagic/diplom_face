import os
from imutils import paths
import shutil
dataset = 'dataset'
output_folder = 'output'


for person in os.listdir(dataset):
    personDir = os.path.join(dataset, person)
    aug_output_folder = os.path.join(personDir, output_folder)
    if os.path.exists(aug_output_folder):
        shutil.rmtree(aug_output_folder)
    personPaths = list(paths.list_images(personDir))
    if len(personPaths) > 1:
        for (i, imagePath) in enumerate(personPaths):
            if i != 1:
                os.remove(imagePath)
