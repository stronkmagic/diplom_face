import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import face_recognition
from matplotlib import pyplot

from PIL import Image
import numpy as np
# random example images
image = face_recognition.load_image_file("../dataset/mydb2/Vladlen/vlad1.jpg")
#images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
images = [image]

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.02), per_channel=0.5), # randomly remove up to 2% of the pixels
                ]),
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
            ],
            random_order=True
        )
    ],
    random_order=True
)

for lp in range(10):
    print(lp)
    image_aug = seq(images=images)
    for img in image_aug:
        im = Image.fromarray(img)
        im.save(str(lp)+'test.png')