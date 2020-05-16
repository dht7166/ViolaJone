import json
import os
import cv2


def read_json(json_path):
    """
    :param json_path: path to json
    :return: json ls
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except:
        data = None
    return data


def json_integrity_check(json_path):
    """
    :param json_path: path to json
    :return:  boolean results
    """
    try:
        data = read_json(json_path)

        # check if exist
        if data is None:
            print("json file does not exist or corrupted!")
            return False

        # check if is list
        if not isinstance(data, list):
            print("json file shall be a list of dictionary, but what you provided is {}!".format(type(data)))
            return False

        # check if key words correct
        for elm_dict in data:
            if not isinstance(elm_dict, dict):
                print("nested inside the list shall be dictionary, not {}".format(type(elm_dict)))
                return False
            if not "iname" in elm_dict:
                print("the nested dictionary miss the key: iname!")
                return False
            if not "bbox" in elm_dict:
                print("the nested dictonary miss the key: bbox!")
                return False
            if not isinstance(elm_dict["iname"], str):
                print("type of the value of image name(iname) is not string")
                return False
            if not isinstance(elm_dict["bbox"], list):
                print("type of the value of bbox is not list!")
                return False
            if len(elm_dict["bbox"]) != 4 :
                print("length of list of bbox shall equal to 4!")
                return False

        print("Your json is good!")
        return True

    except:
        print("the json file is corrupted!")
        return False


def process_json(json_path):
    """
    :param json_path:
    :return: reorganized dict()
    """
    data = read_json(json_path)

    master_dict = dict()
    for elm_dict in data:
        file_name = elm_dict["iname"]
        if not file_name in master_dict:
            master_dict[file_name] = []
        master_dict[file_name].append(elm_dict["bbox"])

    return master_dict


def draw(json_path, img_dir, plot_folder):

    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    master_dict = process_json(json_path)

    for img_name, box_ls in master_dict.items():
        img_full_path = os.path.join(img_dir, img_name)
        if not os.path.isfile(img_full_path):
            continue

        # read img
        img = cv2.imread(img_full_path, cv2.IMREAD_COLOR)

        # annotate
        for box in box_ls:
            c = box[0]
            r =box[1]
            c_w = box[2]
            r_w = box[3]
            img = cv2.rectangle(img, (c, r), (c + c_w, r + r_w), (0, 255, 0), 1)

        # save annotated img
        cv2.imwrite(os.path.join(plot_folder, img_name), img)

def main():
    json_path = os.path.join("json", "results.json")
    img_dir = "imgs"
    plot_folder = "annotated"
    if not json_integrity_check(json_path):
        return
    draw(json_path=json_path, img_dir=img_dir, plot_folder=plot_folder)


if __name__ == "__main__":
    main()





