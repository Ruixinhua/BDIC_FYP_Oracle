import os
import pickle

import numpy as np
import torch
import tools
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import configuration
from image_folder_with_paths import ImageFolderWithPaths


set_iter_no = configuration.set_iter_no
device = configuration.device
root_dir = configuration.dataset_root_dir
data_root_dirs_default = [os.path.join(configuration.cur_data_dir, d) for d in configuration.char_types]
output_file_default = os.path.join("output", "output-%s.pkl" % set_iter_no)


def get_data_loader_with_path(data_dir, batch_size=16, transform=None):
    if not transform:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    dataset = ImageFolderWithPaths(data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return data_loader


def run_model(model, data_loader, model_type="vae"):
    """
    put data into model and get return value of feature vector
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
    return np.array(images_feature), np.array(images_label), np.array(images_path)


def get_model_output(data_root_dirs=None, output_file=output_file_default, model_type="vae",
                     model_paths=configuration.model_paths):
    """
    get model output of specific root directory and return a tuple include feature, label and path
    """
    if data_root_dirs is None:
        data_root_dirs = data_root_dirs_default
    images_results = []
    if os.path.exists(output_file):
        # load result from file
        return pickle.load(open(output_file, "rb"))
    if not os.path.exists("result"):
        os.mkdir("result")
    models = [tools.get_model_by_state(model_path, tools.get_default_model_class(model_type)) for model_path in model_paths]
    transform = tools.get_default_transform(model_type)
    for data_dir, model in zip(data_root_dirs, models):
        model.eval()
        data_loader = get_data_loader_with_path(data_dir, transform=transform)
        # the order of
        model_output = run_model(model, data_loader, model_type)
        images_results.append(model_output)
        print("Shape of images feature", model_output[0].shape)
        print("Shape of images labels", model_output[1].shape)
        print("Shape of images path", model_output[2].shape)
    # get data and save to file
    pickle.dump(images_results, open(output_file, "wb"))
    return images_results

