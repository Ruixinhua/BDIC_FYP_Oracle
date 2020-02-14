import os


for directory in os.scandir("png/"):
    print(directory.name, "Info:")
    print("the total numbers:", len(os.listdir(directory.path)))
    cat = set()
    for file in os.scandir(directory.path):
        cat.add(file.name.split("_")[0][1:])
    print("category:", len(cat))

