import os


def main():
    check_directories = ["base", "data_loader", "logger", "model", "trainer", "dataset"]
    for directory in check_directories:
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                python_file = os.path.join(root, name)
                if python_file.endswith(".py"):
                    os.system("flake8 " + python_file)
            for name in dirs:
                if name == "dataset": continue
                python_file = os.path.join(root, name)
                if python_file.endswith(".py"):
                    os.system("flake8 " + python_file)


if __name__ == "__main__":
    main()

