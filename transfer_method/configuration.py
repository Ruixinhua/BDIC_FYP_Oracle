import torch
import os


def create_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


set_iter_no = 1
run_in_server = False
if not run_in_server:
    if os.name == "posix":
        # 本地没有GPU的话，设置成cpu
        device = torch.device("cpu")
        # 设置本地文件路径
        dataset_root_dir = "/Users/ruixinhua/Documents/pytorch_image_classifier-master/datasets/"
        model_root_dir = None
    else:
        # 设置GPU设备
        cuda_device = "cuda:0"
        device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
        dataset_root_dir = "C:\\Users\\Rui\\Documents\\dataset\\"
        model_root_dir = "C:\\Users\\Rui\\Documents\\BDIC_FYP_Oracle\\transfer_method\\model"
else:
    # 设置GPU设备
    cuda_device = "cuda:0"
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
    # 服务器上的文件路径
    dataset_root_dir = "/home/dairui/data/datasets/"
    model_root_dir = "/home/dairui/workplace/BDIC_FYP_Oracle/transfer_method/target_model/"
set_iter_path = os.path.join(dataset_root_dir, "set_iter%s" % set_iter_no)
# data_dir = set_iter_path
cur_data_dir = os.path.join(dataset_root_dir, "paired_jia_jin")
# chars = ["jia", "jin", "zhuan"]
char_types = ["jia", "jin"]
model_names = ["model_%s_iter-%s" % (c, set_iter_no) for c in char_types]
# model_paths = [os.path.join("target_model", "%s.pkl" % model_name) for model_name in model_names]
cur_model_path = os.path.join("target_model", "model_jin_iter-1.pkl")
model_type = "vae"
model_paths = (os.path.join(model_root_dir, "jia_%s_base_full.pkl" % model_type),
               os.path.join(model_root_dir, "jin_%s_base_full.pkl" % model_type))
# model_paths = [cur_model_path, cur_model_path]
model_reduction_path = os.path.join("reduction", "reduction_result_iter-%s.csv" % set_iter_no)
create_dirs([set_iter_path, "target_model", "reduction", "log", "output", "plot", "cluster"])

