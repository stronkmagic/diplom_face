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
        p = Augmentor.Pipeline(identity_path, output_directory='.')
        # Add operations to the pipeline as normal:
        p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
        p.flip_left_right(probability=0.5)
        p.shear(probability=0.6, max_shear_left=5, max_shear_right=5)
        p.skew_tilt(probability=0.6, magnitude=0.5)
        p.crop_random(probability=1, percentage_area=0.9, randomise_percentage_area=False)
        p.sample(samples)


augment_identity('./dataset/Vladlen/', 10)