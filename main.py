from optparse import OptionParser, OptionGroup
from kvgg import face_recognition_start as k_vgg_face_rec
from dlib_face import start_face_rec
from datetime import datetime


def main():
    parser = OptionParser()
    parser.add_option("-f", "--force", action="store_true", help="Force recompute")
    parser.add_option("-m", "--model", type="string", help="CNN model", default="dlib")
    parser.add_option("-d", "--database", type="string", help="DB name", default="mydb")
    parser.add_option("-a", "--algorithm", type="string", help="Optimization algorithm")

    (options, args) = parser.parse_args()

    stat_file = "./results/"+options.model + "_" + options.database + "_" + datetime.now().strftime('%Y%m%d%H%M%S') + "_result.csv"
    features_files = "./data/" + options.model + "_" + options.database + "_features.pickle"
    if options.model == "dlib":
        start_face_rec(db=options.database, stat_file=stat_file, force_pre_compute=options.force, pre_compute_feature_file=features_files)
    else:
        k_vgg_face_rec(model=options.model, db=options.database, stat_file=stat_file, force_pre_compute=options.force,
                       pre_compute_feature_file=features_files)


if __name__ == "__main__":
    main()
