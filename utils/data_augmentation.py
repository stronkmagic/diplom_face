# USAGE
# python data_augmentation.py --dataset dataset \
#	--samples 10

import Augmentor
# from imutils import paths
# import argparse
import os
# import shutil

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input directory of faces + images")
# ap.add_argument("-s", "--samples", required=True, help="samples for each image in dataset")
#
# args = vars(ap.parse_args())
#
# imagePaths = args["dataset"]
# samples = int(args["samples"])
# imageCount = len(list(paths.list_files(imagePaths)))


def augment_identity(identity_path, samples):
    if os.path.exists(identity_path):
        pipeline = Augmentor.Pipeline(identity_path, output_directory='output')
        #aug_skew(pipeline, samples)
        #aug_distortion(pipeline, samples)
        aug_tilt_x(pipeline, samples)
        aug_tilt_y(pipeline, samples)
        #aug_rotate(pipeline, samples)
        aug_flip_left_right(pipeline, samples)
        #aug_shear(pipeline, samples)


def aug_skew(pipeline, samples):
    pipeline.skew_tilt(probability=1, magnitude=0.2)
    pipeline.sample(samples)


def aug_distortion(pipeline, samples):
    pipeline.random_distortion(probability=1.0, grid_width=4, grid_height=4, magnitude=12)
    pipeline.sample(samples)


def aug_tilt_x(pipeline, samples):
    pipeline.skew_left_right(probability=1, magnitude=0.5)
    pipeline.sample(samples)


def aug_tilt_y(pipeline, samples):
    pipeline.skew_top_bottom(probability=1, magnitude=0.5)
    pipeline.sample(samples)


def aug_rotate(pipeline, samples):
    pipeline.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
    pipeline.sample(samples)


def aug_shear(pipeline, samples):
    pipeline.shear(probability=1, max_shear_left=10, max_shear_right=10)
    pipeline.sample(samples)


def aug_flip_left_right(pipeline, samples):
    pipeline.flip_left_right(probability=1)
    pipeline.sample(samples)


for person in os.listdir('images'):
    personDir = os.path.join('images', person)
    augment_identity(personDir, 3)