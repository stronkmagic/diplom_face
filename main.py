from optparse import OptionParser
from datetime import datetime
from face_rec import start_face_rec_test


def main():
    parser = OptionParser()
    parser.add_option("-f", "--force", action="store_true", help="Force recompute")
    parser.add_option("-m", "--model", type="string", help="CNN model", default="dlib")
    parser.add_option("-d", "--database", type="string", help="DB name", default="mydb")
    parser.add_option("-a", "--alignment", action="store_true", help="Use alignment")
    parser.add_option("-g", "--augmentation", action="store_true", help="Use augmentation")

    (options, args) = parser.parse_args()

    stat_file = "./results/" + options.model + "_" + options.database + "_" + datetime.now().strftime(
        '%Y%m%d%H%M%S') + "_result.csv"
    features_files = "./data/" + options.model + "_" + options.database
    if options.alignment:
        features_files += "_align"
        stat_file += "_align"
    if options.augmentation:
        features_files += "_augm"
        features_files += "_augm"
    features_files += "_features.pickle"

    start_face_rec_test(model=options.model, db=options.database, stat_file=stat_file, force_pre_compute=options.force,
                        features_files=features_files, augm_on=options.augmentation, align_on=options.alignment)


if __name__ == "__main__":
    main()
