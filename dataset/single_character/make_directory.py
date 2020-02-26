import os
import shutil


def mk_dirs(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)


def main():
    # raw data path
    raw_data_dir = "raw_data/"
    # oracle inscriptions path
    oracle_dir = "oracle_inscriptions/"
    # bronze inscriptions path
    bronze_dir = "bronze_inscriptions/"
    # liu shu tong path
    liu_dir = "liu_shu_tong/"
    # shuo wen jie zi path
    shuo_dir = "shuo_wen_jie_zi/"
    print("make directory")
    format_dirs = ["png/", "svg/"]
    type_dirs = [oracle_dir, bronze_dir, liu_dir, shuo_dir]
    mk_dirs(format_dirs)
    for format_dir in format_dirs:
        mk_dirs([format_dir + type_dir for type_dir in type_dirs])
    characters_ls = os.listdir(raw_data_dir)
    characters_dic = {}
    index = 0
    # read and save characters as a dictionary
    with open("dictionary.txt", "w", encoding="utf8") as w:
        for c in characters_ls:
            characters_dic[c] = index
            w.write(c + "\n")
            index += 1
    # create a path mapping of dataset
    path_mapping = {"甲骨文": oracle_dir, "金文": bronze_dir, "六书通": liu_dir, "说文解字的篆字": shuo_dir}
    for chara_dir in os.scandir(raw_data_dir):
        # find the index of this character in dictionary
        file_no = characters_dic[chara_dir.name]
        if not chara_dir.is_dir():
            continue
        for type_dir in os.scandir(chara_dir.path):
            if type_dir.name not in path_mapping:
                continue
            sub_dir = path_mapping[type_dir.name]
            for format_dir in os.scandir(type_dir.path):
                if not format_dir.is_dir():
                    continue
                file_index = 1
                for old_file in os.scandir(format_dir.path):
                    old_file_name = old_file.name
                    # new file path with format: typePath/J(B/L/S)1_1.png
                    new_file_path = "%s/%s%s%s_%s%s" % (format_dir.name, sub_dir, old_file_name[0], file_no, file_index,
                                                        old_file_name[-4:])
                    print("%s =>> %s" % (old_file.path, new_file_path))
                    shutil.copyfile(old_file.path, new_file_path)
                    file_index += 1


if __name__ == "__main__":
    main()
