import torch
import os
import tools


class Configuration:

    def _set_device(self, device_id=0):
        # set GPU device
        self.device_id = device_id
        self.device = torch.device("cuda:%s" % device_id if torch.cuda.is_available() else "cpu")

    def __init__(self, device_id=0, model_type="VanillaVAE", paired_chars=None, dataset_type="cluster"):
        if paired_chars is None:
            paired_chars = ["jia", "jin"]
        self.paired_chars = paired_chars
        self.dataset_type = dataset_type
        self._set_device(device_id=device_id)
        self.model_type = model_type
        self.saved_path, self.log_file, self.best_model_path = None, None, None
        self.set_path(os.path.join("_".join(self.paired_chars), model_type, dataset_type))

    def set_path(self, path):
        self.saved_path = os.path.join("checkpoint", path)
        log_path = os.path.join("log", path)
        tools.make_dir(self.saved_path)
        tools.make_dir(log_path)
        self.log_file = os.path.join(log_path, "log.txt")
        self.best_model_path = os.path.join(self.saved_path, "model_best.pth")


class ModelConfiguration(Configuration):

    def __init__(self, device_id=0, model_type="VanillaVAE", criterion="mmd", strategy="single", learning_rate=1e-3,
                 epochs=100, early_stop=30, save_period=10, stage="base", model_params=None, paired_chars=None,
                 mode="instance", dataset_type="cluster"):
        super().__init__(device_id, model_type, paired_chars, dataset_type)
        self.mode, self.criterion, self.strategy, self.lr = mode, criterion, strategy, learning_rate
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
        saved_path = os.path.join("_".join(self.paired_chars), model_type, mode, dataset_type,
                                  "_".join([str(p) for p in model_params.values()]))
        self.set_path(saved_path)
