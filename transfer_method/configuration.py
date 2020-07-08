import torch
import os


class Configuration:

    def __init__(self, device_id=0, model_type="VanillaVAE", criterion="mmd", strategy="single", learning_rate=1e-3,
                 epochs=100, early_stop=30, save_period=10, stage="base", model_params=None, paired_chars=None,
                 mode="instance"):
        if paired_chars is None:
            paired_chars = ["jia", "jin"]
        self.paired_chars, self.mode = paired_chars, mode
        self._set_device(device_id=device_id)
        self.model_type, self.criterion, self.strategy, self.lr = model_type, criterion, strategy, learning_rate
        self.epochs, self.early_stop, self.save_period, self.stage = epochs, early_stop, save_period, stage
        if model_params is None:
            if model_type == "AE":
                model_params = {"input_shape": 96 * 96}
            elif model_type == "VanillaVAE":
                model_params = {"input_size": 96}
            elif model_type == "ResNet_VAE":
                model_params = {"fc_hidden1": 1024, "fc_hidden2": 1024, "CNN_embed_dim": 256}
            elif model_type == "VQVAE":
                model_params = {"embedding_dim": 96, "num_embeddings": 256}

        self.model_params = model_params
        self.saved_path = os.path.join("checkpoint", "_".join(self.paired_chars), model_type, mode,
                                       "_".join([str(p) for p in model_params.values()]))
        log_path = os.path.join("log", "_".join(self.paired_chars), model_type, mode)
        create_dirs([self.saved_path, log_path])
        self.log_file = os.path.join(log_path, "%s.txt" % ("_".join([str(p) for p in model_params.values()])))
        self.best_model_path = os.path.join(self.saved_path, "model_best.pth")

    def _set_device(self, device_id=0):
        # set GPU device
        self.device_id = device_id
        self.device = torch.device("cuda:%s" % device_id if torch.cuda.is_available() else "cpu")


def create_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


