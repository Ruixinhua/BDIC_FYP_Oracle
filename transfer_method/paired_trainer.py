# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/4/28 17:45
# @Function      : This is the class of paired trainer
from abc import abstractmethod

import torch
import torch.optim as optim
import os
import tools
from configuration import ModelConfiguration
import model_helper
import loss_helper
from paired_dataset import PairedDataset
from prediction import Prediction

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class BasePairedTrainer:
    """
    Base class for paired dataset training
    """

    def __init__(self, config: ModelConfiguration, dataset: PairedDataset):
        self.config = config
        # setup GPU device if available, move models into configured device
        self.device = self.config.device
        self.log_file = open(self.config.log_file, "a+", encoding="utf-8")
        tools.print_log("Using %s!!!" % self.device, file=self.log_file)

        # init models
        self.target_model, self.target_optimizer = self._get_model_opt()
        if self.config.strategy == "both":
            self.source_model, self.source_optimizer = self._get_model_opt()
        else:
            self.source_model, self.source_optimizer = self.target_model, self.target_optimizer
        self.model_type, self.criterion = self.config.model_type, self.config.criterion
        self.start_epoch, self.best_acc, self.best_index_sum = 1, 0.0, 0
        # init dataset
        self.dataset = dataset
        self.target_data, self.source_data, self.labels = dataset.target_data, dataset.source_data, dataset.labels
        self.prediction_result = []
        self.add_cons = True if self.config.stage == "add" else False
        tools.print_log("Load data success!!!", file=self.log_file)

    def _get_model_opt(self):
        model = tools.get_default_model_class(self.config.model_type, **self.config.model_params).to(self.config.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        return model, optimizer

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to "model_best.pth"
        """
        state = {
            "best_acc": self.best_acc, "best_index_sum": self.best_index_sum, "config": self.config, "epoch": epoch,
            "target_model": self.target_model.state_dict(), "target_optimizer": self.target_optimizer.state_dict(),
        }
        if self.target_model != self.source_model:
            state["source_model"] = self.source_model.state_dict()
            state["source_optimizer"] = self.source_optimizer.state_dict()
        filename = os.path.join(self.config.saved_path, "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)
        tools.print_log("Saving checkpoint: {} ...".format(filename), file=self.log_file)
        if save_best:
            best_path = os.path.join(self.config.saved_path, "model_best.pth")
            torch.save(state, best_path)
            tools.print_log("Saving current best: model_best.pth ...", file=self.log_file)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        tools.print_log("Loading checkpoint: {} ...".format(resume_path), file=self.log_file)
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.config = checkpoint["config"] if "config" in checkpoint else self.config
        # load accuracy of checkpoint
        self.best_acc = checkpoint["best_acc"]
        self.best_index_sum = checkpoint["best_index_sum"]

        # load architecture params from checkpoint.
        self.target_model, self.target_optimizer = self._get_model_opt()
        self.target_model.load_state_dict(checkpoint["target_model"])
        if "source_model" in checkpoint:
            self.source_model, self.source_optimizer = self._get_model_opt()
            self.source_model.load_state_dict(checkpoint["source_model"])
        else:
            self.source_model = self.target_model

        # load optimizer state from checkpoint only when target_optimizer type is not changed.
        self.target_optimizer.load_state_dict(checkpoint["target_optimizer"])
        if "source_optimizer" in checkpoint:
            self.source_optimizer.load_state_dict(checkpoint["source_optimizer"])
        else:
            self.source_optimizer = self.target_optimizer

        tools.print_log("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch),
                        file=self.log_file)

    @abstractmethod
    def _train_epoch(self, **kwargs):
        """
        Training logic for an epoch

        Returns: A dictionary of log that will be output

        """
        raise NotImplementedError

    @staticmethod
    def _backward(opt, loss):
        # reset the gradients back to zero
        opt.zero_grad()
        # PyTorch accumulates gradients on subsequent backward passes
        loss.backward()
        opt.step()

    def train(self, resume_path=None):
        """
        Full training logic
        """
        if resume_path is not None:
            self._resume_checkpoint(resume_path)
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            log = {"epoch": "%s/%s" % (epoch, self.config.epochs)}
            self.target_model.train()
            self.source_model.train()
            # save logged information into log dict
            log.update(self._train_epoch())
            best = False
            self.target_model.eval()
            self.source_model.eval()
            pred = Prediction(self.dataset.target_val, self.dataset.labels_val, self.dataset.source_data_full,
                              self.dataset.source_labels_full, self.dataset.paths_val, model_type=self.model_type)
            val_result, val_paths = pred.predict(self.target_model, self.source_model)
            self.prediction_result.append([val_result, val_paths])
            log.update(val_result)
            if self.dataset.source_data_exp is not None and self.dataset.source_labels_exp is not None:
                pred.set_source(self.dataset.source_data_exp, self.dataset.source_labels_exp, set_type="val_exp")
                exp_result, exp_paths = pred.predict(self.target_model, self.source_model)
                self.prediction_result.append([exp_result, exp_paths])
                log.update(exp_result)
            # evaluate target_model performance according to configured metric, save best checkpoint as model_best
            cur_acc, cur_index_sum = val_result["val_accuracy"], val_result["val_index_sum"]
            # check whether models performance improved or not
            improved = (cur_acc > self.best_acc) or \
                       (cur_acc == self.best_acc and cur_index_sum < self.best_index_sum)

            if improved:
                not_improved_count, self.best_acc, self.best_index_sum, best = 0, cur_acc, cur_index_sum, True
                self._save_checkpoint(epoch, save_best=best)
            else:
                not_improved_count += 1

            # print logged information to the screen
            for key, value in log.items():
                tools.print_log("{:30s}: {}".format(str(key), value), file=self.log_file)

            if not_improved_count > self.config.early_stop:
                tools.print_log("Validation performance did not improve for %s epochs.So Stop" % self.config.early_stop,
                                file=self.log_file)
                break

            if epoch % self.config.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)


class PairedTrainer(BasePairedTrainer):

    def _train_epoch(self, **kwargs):
        target_recon_loss, source_recon_loss, dis_loss_all, count = 0.0, 0.0, 0.0, 0.0
        target_data, source_data, labels = self.dataset.random_data(self.target_data, self.source_data, self.labels)
        for target_batch, source_batch, label_batch in zip(target_data, source_data, labels):
            target_batch, source_batch = target_batch.to(self.config.device), source_batch.to(self.config.device)
            if not self.add_cons:
                target_code, target_loss = model_helper.run_batch(self.target_model, target_batch, self.model_type)
                self._backward(self.target_optimizer, target_loss)
                source_code, source_loss = model_helper.run_batch(self.source_model, source_batch, self.model_type)
                self._backward(self.source_optimizer, source_loss)
                dis_loss = loss_helper.cal_dis_err(target_code, source_code, label_batch, criterion=self.criterion)
            else:
                if self.target_model != self.source_model:
                    target_code, target_loss = model_helper.run_batch(self.target_model, target_batch, self.model_type)
                    source_code, _ = model_helper.run_batch(self.source_model, source_batch, self.model_type,
                                                            train=False)
                    dis_loss = loss_helper.cal_dis_err(target_code, source_code, label_batch, criterion=self.criterion)
                    combined_loss = target_loss + dis_loss
                    self._backward(self.target_optimizer, combined_loss)

                    target_code, _ = model_helper.run_batch(self.target_model, target_batch, self.model_type,
                                                            train=False)
                    source_code, source_loss = model_helper.run_batch(self.source_model, source_batch, self.model_type)
                    dis_loss = loss_helper.cal_dis_err(target_code, source_code, label_batch, criterion=self.criterion)
                    combined_loss = source_loss + dis_loss
                    self._backward(self.source_optimizer, combined_loss)
                else:
                    # suppose target models is equal to source models
                    target_code, target_loss = model_helper.run_batch(self.target_model, target_batch, self.model_type,
                                                                      device=self.device)
                    source_code, source_loss = model_helper.run_batch(self.source_model, source_batch, self.model_type,
                                                                      device=self.device)
                    # add weight here
                    target_weight, source_weight, dis_weight = 1, 1, 1
                    dis_loss = loss_helper.cal_dis_err(target_code, source_code, label_batch, criterion=self.criterion)
                    combined_loss = target_weight * target_loss + source_weight * source_loss + dis_weight * dis_loss
                    self._backward(self.target_optimizer, combined_loss)

            # calculate loss here
            count += len(label_batch)
            target_recon_loss += target_loss.item()
            source_recon_loss += source_loss.item()
            dis_loss_all += dis_loss.item()
        return {"Target reconstruct loss": target_recon_loss / count,
                "Source reconstruct loss": source_recon_loss / count, "Distance loss": dis_loss_all / count}


class GradNormTrainer(BasePairedTrainer):
    def __init__(self, config: ModelConfiguration, dataset: PairedDataset):
        super().__init__(config, dataset)
        self.target_weight = torch.tensor([1.], requires_grad=True)
        self.source_weight = torch.tensor([1.], requires_grad=True)
        self.dis_weight = torch.tensor([1.], requires_grad=True)
        self.weight_opt = torch.optim.Adam([self.target_weight, self.source_weight, self.dis_weight], lr=1e-3)
        # self.target_weight = self.target_weight.to(self.device)
        # self.source_weight = self.source_weight.to(self.device)
        # self.dis_weight = self.dis_weight.to(self.device)
        self.grad_loss = torch.nn.L1Loss()
        self.alpha = 0.16
        self.l0 = None
        self.isFirst = True

    def _train_epoch(self, **kwargs):
        target_recon_loss, source_recon_loss, dis_loss_all, count = 0.0, 0.0, 0.0, 0.0
        target_data, source_data, labels = self.dataset.random_data(self.target_data, self.source_data, self.labels)
        for target_batch, source_batch, label_batch in zip(target_data, source_data, labels):
            target_batch, source_batch = target_batch.to(self.device), source_batch.to(self.device)
            target_code, target_loss = model_helper.run_batch(self.target_model, target_batch, self.model_type, device=self.device)
            source_code, source_loss = model_helper.run_batch(self.source_model, source_batch, self.model_type, device=self.device)
            dis_loss = loss_helper.cal_dis_err(target_code, source_code, label_batch, criterion=self.criterion)
            # add weight here
            target_loss = torch.mul(self.target_weight, target_loss.cpu())
            source_loss = torch.mul(self.source_weight, source_loss.cpu())
            dis_loss = torch.mul(self.dis_weight, dis_loss.cpu())
            total_loss = (target_loss + source_loss + dis_loss)
            if self.isFirst:
                self.l0 = torch.div(total_loss, 3)
                self.isFirst = False

            self.target_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            tools.print_log('Total loss: %s' % total_loss.item(), file=self.log_file)

            # Calculating relative losses
            lhat_source = torch.div(source_loss, self.l0)
            lhat_target = torch.div(target_loss, self.l0)
            lhat_dis = torch.div(dis_loss, self.l0)
            lhat_avg = torch.div((lhat_source + lhat_target + lhat_dis), 3)

            # Calculating relative inverse training rates for tasks
            rate_source = torch.div(lhat_source, lhat_avg)
            rate_target = torch.div(lhat_target, lhat_avg)
            rate_dis = torch.div(lhat_dis, lhat_avg)

            # Getting gradients of the first layers of each tower and calculate their l2-norm
            param = list(self.target_model.cpu().parameters())
            GR_source = torch.autograd.grad(source_loss, param[0], retain_graph=True, create_graph=True)
            G_source = torch.norm(GR_source[0], 2)
            GR_target = torch.autograd.grad(target_loss, param[0], retain_graph=True, create_graph=True)
            G_target = torch.norm(GR_target[0], 2)
            GR_dis = torch.autograd.grad(dis_loss, param[0], retain_graph=True, create_graph=True, allow_unused=True)
            G_dis = torch.norm(GR_dis[0], 2)
            G_avg = torch.div((G_source + G_target + G_dis), 3)

            # Calculating the constant target for Eq. 2 in the GradNorm paper
            C_source = (G_avg * (rate_source) ** self.alpha).detach()
            C_target = (G_avg * (rate_target) ** self.alpha).detach()
            C_dis = (G_avg * (rate_dis) ** self.alpha).detach()

            self.weight_opt.zero_grad()
            grad_loss = self.grad_loss(G_source, C_source) + self.grad_loss(G_target, C_target) + self.grad_loss(G_dis,
                                                                                                                 C_dis)
            tools.print_log('grad_loss: %s' % (grad_loss.item()), file=self.log_file)
            grad_loss.backward(create_graph=True)
            self.weight_opt.step()

            self.target_optimizer.step()

            # Renormalizing the losses weights
            coef = 3 / (self.source_weight + self.target_weight + self.dis_weight).item()
            self.source_weight = self.source_weight * coef
            self.target_weight = self.target_weight * coef

            self.dis_weight = self.dis_weight * coef
            tools.print_log('source weight:%s, target weight:%s, dis weight:%s' %
                            (self.source_weight[0], self.target_weight[0], self.dis_weight[0]), file=self.log_file)

            # calculate loss here
            count += len(label_batch)
            target_recon_loss += target_loss.item()
            source_recon_loss += source_loss.item()
            dis_loss_all += dis_loss.item()
        return {"Target reconstruct loss": target_recon_loss / count,
                "Source reconstruct loss": source_recon_loss / count, "Distance loss": dis_loss_all / count}


if __name__ == "__main__":
    # Basic test code here, using VanillaVAE model as test model
    model_type, criterion, strategy, stage, mode = "VanillaVAE", "mmd", "single", "add", "instance"
    paired_chars = ["jia", "jin"]
    model_params = {"embedding_dim": 16, "num_embeddings": 512}
    dataset_type = "cluster"
    conf = ModelConfiguration(device_id=0, model_type=model_type, criterion=criterion, strategy=strategy, stage=stage,
                              paired_chars=paired_chars, mode=mode, dataset_type=dataset_type)
    tools.print_log(conf.model_params, file=open(conf.log_file, "a+"))
    paired_dataset = PairedDataset(False, conf=conf)
    paired_dataset.split_dataset()
    # change path to which models you want resume from
    checkpoint_path = "checkpoint/%s_base_one/checkpoint-epoch60.pth" % model_type
    test_trainer = PairedTrainer(config=conf, dataset=paired_dataset)
    test_trainer.train()
