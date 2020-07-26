import torch
import os
from torchvision import transforms
import torch.optim as optim
import shutil
from datetime import datetime


seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def print_log(info_str, other_info='', file=None):
    current_time = datetime.now().strftime('%Y-%m-%c %H:%M:%S')
    if file:
        print("%s [INFO]: %s %s" % (current_time, info_str, other_info), file=file)
        file.flush()
    else:
        print("%s [INFO]: %s %s" % (current_time, info_str, other_info))


def copy_files(source_paths, des_paths, is_debug=False):
    """
    将源文件移到目标文件夹
    """
    for source_path, des_path in zip(source_paths, des_paths):
        if not os.path.exists(os.path.dirname(des_path)):
            os.makedirs(os.path.dirname(des_path))
        shutil.copyfile(source_path, des_path)
        if is_debug:
            print_log("Copy file from %s to %s" % (source_path, des_path))


def get_device(device_id=0):
    """
    setup GPU device if available, move models into configured device
    """
    if torch.cuda.is_available():
        return torch.device("cuda:%d" % device_id)
    else:
        return torch.device("cpu")


def get_default_model_class(model_type="AE", **model_params):
    model_object = __import__("models")
    model_class = getattr(model_object, model_type)
    return model_class(**model_params)


def get_default_transform(model_type, img_size=96):
    if model_type == "AE":
        return transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    elif model_type == "ResNet_VAE":
        return transforms.Compose([transforms.Resize([img_size, img_size]), transforms.ToTensor()])
    elif model_type == "VanillaVAE":
        return transforms.Compose([transforms.Resize([img_size, img_size]), transforms.ToTensor()])
    else:
        return transforms.Compose([transforms.Resize([img_size, img_size]), transforms.ToTensor()])


def get_model_by_state(state_dic_path, model_class, device=get_device()):
    if state_dic_path is not None and os.path.exists(state_dic_path):
        model_class.load_state_dict(torch.load(state_dic_path, map_location=device))
    return model_class


def get_model_opt(state_dic_path, model_class, learning_rate=1e-3, device=get_device()):
    model = get_model_by_state(state_dic_path, model_class, device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

