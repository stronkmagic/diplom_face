import imgaug.augmenters as iaa

from matplotlib import pyplot as plt
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
                iaa.Add((-50, 50), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.LinearContrast((0.5, 3.0), per_channel=0.5), # improve or worsen the contrast
            ],
            random_order=True
        )
    ],
    random_order=True
)


def face_augmentation(image, num_samples=10):
    resultImages = []
    for lp in range(num_samples):
        images = seq(images=[image])
        if len(images) > 0:
            resultImages.append(images[0])
    return resultImages


# EXAMPLE FOR PAPER PART
# import face_recognition
# image = face_recognition.load_image_file('../dataset/solo/Vladlen/vladlen1.jpg')
# images = [image]
# fig = plt.figure(figsize=(5, 5))
# for lp in range(10):
#     print(lp)
#     image_aug = seq(images=images)
#     fig.add_subplot(2, 5, lp + 1)
#     plt.imshow(image_aug[0])
# plt.show()
