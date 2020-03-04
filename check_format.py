import os


def main():
    # for directory in check_directories:
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            file = os.path.join(root, name)
            if file.endswith(".py"):
                print("check", file)
                os.system("flake8 " + file)


if __name__ == "__main__":
    main()

