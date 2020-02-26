import os


def mk_dirs(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)


def main():
    # raw data path
    raw_data_dir = "single_character/raw_data/"
    format_dirs = ["png"]
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
    paths = {"甲骨文", "金文", "六书通", "说文解字的篆字"}
    file2char = {}
    output = open("file_mapping.txt", "w")
    for chara_dir in os.scandir(raw_data_dir):
        if not chara_dir.is_dir():
            continue
        for type_dir in os.scandir(chara_dir.path):
            if type_dir.name not in paths:
                continue
            for format_dir in os.scandir(type_dir.path):
                if not format_dir.is_dir() or format_dir.name not in format_dirs:
                    continue
                for old_file in os.scandir(format_dir.path):
                    old_file_name = old_file.name
                    if old_file_name not in file2char:
                        file2char[old_file_name] = []
                    file2char[old_file_name].append(chara_dir.name)
    same_char = dict()
    for file_name, chars in file2char.items():
        if len(chars) > 1:
            tmp = tuple(chars)
            if tmp not in same_char:
                same_char[tmp] = []
            same_char[tmp].append(file_name)
    for chars, file_name in same_char.items():
        print(chars, file_name, file=output)


if __name__ == "__main__":
    main()
