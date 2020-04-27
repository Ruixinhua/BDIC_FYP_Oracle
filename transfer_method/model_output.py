import os
import pickle

import pandas as pd
import torch
import tools
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import configuration
from image_folder_with_paths import ImageFolderWithPaths


def get_data_loader_with_path(data_dir, batch_size=16, transform=None):
    """
    iterate data with image path

    :param data_dir: The directory of data
    :param batch_size: size of batch, default is 16
    :param transform: transform of image, default is grayscale and to tensor
    :return: A DataLoader object with image tensor, image label, and image path
    """
    if not transform:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    dataset = ImageFolderWithPaths(data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, )
    return data_loader


def run_model(model, data_loader, model_type="vae", device=tools.get_device()):
    """
    put data into model and get return value of feature vector

    :param model: model object
    :param data_loader: DataLoader object
    :param model_type: the type of model
    :param device: the target device
    :return:
    """
    images_feature, images_label, images_path = [], [], []
    for data in data_loader:
        # 0: image tensor, 1: labels, 2: image path
        images = data[0].to(device=device)
        images_label.extend([data_loader.dataset.classes[label] for label in data[1]])
        # get path from override ImageFolder class
        images_path.extend(data[2])
        # code is a tensor
        code, _ = tools.run_batch(model, images, model_type=model_type, train=False)
        images_feature.extend(code.data.cpu().numpy())
    return images_feature, images_label, images_path


def get_model_output_df(data_root_dirs=None, output_file="output/vae_base_two_1.pkl", model_type="vae", begin=100,
                        end=200, models=None, debug=False, device=tools.get_device()):
    """
    get model output of specific root directory and return a tuple include feature, label and path

    :param data_root_dirs: The directory of data
    :param output_file: The file where the result is stored in
    :param model_type: The type of model, can be "vae", ”ae"
    :param begin: The begin index of char data
    :param end: The end index of char data
    :param models: A list of model,
    :param debug: If True, then it will not load the output_file; default is False
    :param device: The cuda device
    :return: A pandas DataFrame with columns "feature", "label", "path", "type"
    """
    if data_root_dirs is None:
        data_root_dirs = [os.path.join(configuration.cur_data_dir, d) for d in configuration.char_types]
    if models is None:
        models = [tools.get_model_by_state(path, tools.get_default_model_class(model_type))
                  for path in configuration.model_paths]
    if os.path.exists(output_file) and not debug:
        # load result from file
        return pickle.load(open(output_file, "rb"))
    if not os.path.exists("result"):
        os.mkdir("result")
    transform = tools.get_default_transform(model_type)
    char_list = pickle.load(open("char_list.pkl", "rb"))
    model_outputs = pd.DataFrame({"feature": [], "label": [], "path": [], "type": []})
    for data_dir, model in zip(data_root_dirs, models):
        char_type = data_dir.split(os.sep)[-1]
        model.eval()
        for char_dir in char_list[begin:end]:
            data_loader = get_data_loader_with_path(os.path.join(data_dir, char_dir), transform=transform,
                                                    batch_size=500)
            # the order of output is feature, label and path
            output = run_model(model, data_loader, model_type, device=device)
            model_output = {"feature": output[0], "label": [char_dir for _ in range(len(output[1]))], "path": output[2],
                            "type": [char_type for _ in range(len(output[1]))]}
            model_output = pd.DataFrame(model_output)
            model_output["file_name"] = model_output.path.apply(lambda p: p.split(os.sep)[-1])
            model_outputs = model_outputs.append(model_output, ignore_index=True)
    print("Get model output:", model_outputs.shape)
    # get data and save to file
    pickle.dump(model_outputs, open(output_file, "wb"))
    return model_outputs


if __name__ == "__main__":
    # test code
    model_type = "vae"
    model_paths = ("model/jia_%s_base_no_kld.pkl" % model_type, "model/jin_%s_base_no_kld.pkl" % model_type)
    test_models = [tools.get_model_by_state(path, tools.get_default_model_class(model_type)) for path in model_paths]
    output_df = get_model_output_df(debug=True, models=test_models)
