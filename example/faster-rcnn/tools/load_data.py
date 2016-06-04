from helper.dataset.pascal_voc import PascalVOC


def load_voc(image_set, year, root_path, devkit_path, flip=False):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    return voc

