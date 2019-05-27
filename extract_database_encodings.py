# USAGE
# python extract_database_encodings.py --dataset dataset --embeddings output/embeddings.pickle

# import the necessary packages
import timeit
from imutils import paths
import argparse
import pickle
import face_recognition
from multiprocessing import Manager, cpu_count, Pool, Process

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True, help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))


def rename(name):
    if name.find('output') == -1:
        ret_name = name[name.find("/")+1:name.rfind("/")] + " input"
    else:
        ii = name.find('jpg_')
        img_symb = name[ii+4:ii+9]
        name = name.replace('/output', '')
        ret_name = name[name.find("/")+1:name.rfind("/")] + " "+ img_symb
    return ret_name


def process_image(imagePath):
    # Load a sample picture and learn how to recognize it.
    image = face_recognition.load_image_file(imagePath)

    # Get locations and encodings
    face_location = face_recognition.face_locations(image)
    face_encoding = face_recognition.face_encodings(image, face_location, num_jitters=1)
    return face_encoding, rename(imagePath)


if (__name__ == '__main__'):
    # get cpu count
    cpu_count = cpu_count()

    known_face_encodings = []
    known_face_names = []
    total = 0

    chunks = [imagePaths[i:i + cpu_count] for i in range(0, len(imagePaths), cpu_count)]
    pool = Pool(cpu_count)
    print("[INFO] Chunk size {}".format(len(chunks[0])))
    for (iteration, imageChunk) in enumerate(chunks):
        t1 = timeit.default_timer()
        print("[INFO] processing image chunk {}/{}".format(iteration+1, len(chunks)))
        with Pool(cpu_count) as pool_process:
            pool_result = pool_process.map(process_image, imageChunk)
            for single_result in pool_result:
                (face_encoding, name) = single_result
                if len(face_encoding) >= 1:
                    known_face_names.append(name)
                    known_face_encodings.append(face_encoding[0])
                    total += 1

        t2 = timeit.default_timer()
        print("[INFO] Chunk processed in  {} seconds".format(round(t2-t1, 2)))

    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"known_face_encodings": known_face_encodings, "known_face_names": known_face_names}
    f = open(args["embeddings"], "wb")
    f.write(pickle.dumps(data))
    f.close()
