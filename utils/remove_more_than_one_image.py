import os
from imutils import paths


def remove_more_than_one():
    dataset = '../dataset'

    for person in os.listdir(dataset):
        personDir = os.path.join(dataset, person)
        personPaths = list(paths.list_images(personDir))
        if len(personPaths) > 1:
            for (i, imagePath) in enumerate(personPaths):
                if i != 1:
                    os.remove(imagePath)
