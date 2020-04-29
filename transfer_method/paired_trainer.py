# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/4/28 17:45
# @Function      : This is the class of paired trainer
from abc import abstractmethod

import torch
import os
import tools
import data_helper
import model_helper
import loss_helper
import numpy as np


class BasePairedTrainer:
    """
    Base class for paired dataset training
    """
    def __init__(self, target_model, target_optimizer, criterion, source_model=None, source_optimizer=None, device_id=0,
                 epochs=100, early_stop=20, saved_dir="checkpoint/ae_base/", save_period=10, chars=None, data_dir=None,
                 val_num=None, test_num=None, level="all", transform=None, batch_size=16, log_file="log/tmp.txt"):
        # setup GPU device if available, move model into configured device
        if os.path.exists(log_file):
            self.log_file = open(log_file, "a+", encoding="utf-8")
        else:
            self.log_file = open(log_file, "w", encoding="utf-8")
        self.device = self._prepare_device(device_id)

        # init model
        self.target_model, self.target_optimizer = target_model.to(self.device), target_optimizer
        self.source_model = self.target_model if source_model is None else source_model
        self.source_optimizer = self.target_optimizer if source_optimizer is None else source_optimizer
        self.model_type, self.criterion = type(self.target_model).__name__, criterion
        tools.make_dir(saved_dir)
        self.checkpoint_dir, self.early_stop, self.save_period, self.epochs = saved_dir, early_stop, save_period, epochs
        self.start_epoch, self.best_acc, self.best_index_sum = 1, 0.0, 0.0

        # init dataset
        self.chars = ["jia", "jin"] if chars is None else chars
        self.transform = tools.get_default_transform(self.model_type) if transform is None else transform
        self.batch_size, self.level = batch_size, level
        self.add_cons = True if test_num is not None and val_num is not None else False
        # get all paired data
        paired = data_helper.get_paired_data(chars, 0, level, labeled=True, transform=self.transform, data_dir=data_dir)
        self.target_data, self.source_data, self.labels = np.array(paired[0]), np.array(paired[1]), np.array(paired[2])
        if test_num is not None and val_num is not None:
            # target dataset and source dataset share the same labels list
            char_list = torch.load("char_list.pkl")
            # full source data used by predict process
            self.source_full = [torch.tensor(self.source_data[self.labels == c]) for c in char_list]
            self.source_labels = [c for c in char_list]
            # validation and test target set is group by character type, and it is batch data
            self.target_test, self.labels_test = self._split_target(char_list[:test_num])
            self.target_val, self.labels_val = self._split_target(char_list[test_num:val_num + test_num])
            # here training is not batch data
            self.target_train, self.source_train, self.labels_train = self._split_data(char_list[val_num + test_num:])

        else:
            # without test_num and val_num, it is a non-supervised learning, use all data
            self.target_train, self.source_train, self.labels_train = self.target_data, self.source_data, self.labels
        # delete origin dataset for saving memory, as it is just used for temporary
        del(self.target_data, self.source_data, self.labels)
        # get the random batch data of training
        self.target_batches, self.source_batches, self.label_batches = data_helper.random_data(
            self.target_train, self.source_train, self.labels_train, self.batch_size, self.level)

    def _split_data(self, char_list):
        """ Split data by character type, and it is not batch data"""
        target, source, labels = [], [], []
        for char in char_list:
            index = (self.labels == char)
            target.extend(self.target_data[index])
            source.extend((self.source_data[index]))
            labels.extend([char for _ in range(sum(index))])
        return target, source, labels

    def _split_target(self, char_list):
        """ Split target into a list of batch """
        target = [torch.tensor(self.target_data[self.labels == c]) for c in char_list]
        labels = [c for c in char_list]
        return target, labels

    def _prepare_device(self, device_id):
        """
        setup GPU device if available, move model into configured device
        """
        if torch.cuda.is_available():
            tools.print_log("Using GPU %d!!!" % device_id, file=self.log_file)
            return torch.device("cuda:%d" % device_id)
        else:
            tools.print_log("No GPU available, using CPU", file=self.log_file)
            return torch.device("cpu")

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to "model_best.pth"
        """
        model_type = type(self.target_model).__name__
        state = {
            "model_type": model_type, "epoch": epoch, "best_acc": self.best_acc, "best_index_sum": self.best_index_sum,
            "target_model": self.target_model.state_dict(), "target_optimizer": self.target_optimizer.state_dict(),
        }
        if self.target_model != self.source_model:
            state["source_model"] = self.source_model.state_dict()
            state["source_optimizer"] = self.source_optimizer.state_dict()
        filename = os.path.join(self.checkpoint_dir, "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)
        tools.print_log("Saving checkpoint: {} ...".format(filename), file=self.log_file)
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
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
        # load accuracy of checkpoint
        self.best_acc = checkpoint["best_acc"]
        self.best_index_sum = checkpoint["best_index_sum"]

        # load architecture params from checkpoint.
        self.target_model.load_state_dict(checkpoint["target_model"])
        if "source_model" in checkpoint:
            self.source_model.load_state_dict(checkpoint["source_model"])
        else:
            self.source_model = self.target_model

        # load optimizer state from checkpoint only when target_optimizer type is not changed.
        self.target_optimizer.load_state_dict(checkpoint["target_optimizer"])
        if "source_optimizer" in checkpoint:
            self.source_optimizer.load_state_dict(checkpoint["source_optimizer"])
        else:
            self.source_optimizer = self.target_optimizer

        tools.print_log("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch), file=self.log_file)

    @abstractmethod
    def _train_epoch(self, **kwargs):
        """
        Training logic for an epoch

        Args:
            epoch: Current epoch number

        Returns:

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
        for epoch in range(self.start_epoch, self.epochs + 1):
            log = {"epoch": "%s/%s" % (epoch, self.epochs)}
            result = self._train_epoch()
            # save logged information into log dict
            log.update(result)
            self.target_batches, self.source_batches, self.label_batches = data_helper.random_data(
                self.target_train, self.source_train, self.labels_train, self.batch_size, self.level)
            best = False
            if self.add_cons:
                val_result = model_helper.predict(
                    self.target_val, self.labels_val, self.source_full, self.source_labels, self.target_model,
                    self.source_model, model_type=self.model_type, criterion=self.criterion
                )
                log.update(val_result)
                # evaluate target_model performance according to configured metric, save best checkpoint as model_best
                if self.add_cons:
                    cur_acc, cur_index_sum = val_result["Accuracy"], val_result["Sum of index"]
                    # check whether model performance improved or not
                    improved = (cur_acc > self.best_acc) or \
                               (cur_acc == self.best_acc and cur_index_sum < self.best_index_sum)

                    if improved:
                        not_improved_count, self.best_acc, self.best_index_sum, best = 0, cur_acc, cur_index_sum, True
                        test_result = model_helper.predict(
                            self.target_test, self.labels_test, self.source_full, self.source_labels, self.target_model,
                            self.source_model, model_type=self.model_type, criterion=self.criterion
                        )
                        log.update({"Model is improved": "The result in test dataset"})
                        log.update(test_result)
                    else:
                        not_improved_count += 1

            # print logged information to the screen
            for key, value in log.items():
                tools.print_log("{:30s}: {}".format(str(key), value), file=self.log_file)

            if not_improved_count > self.early_stop:
                tools.print_log("Validation performance did not improve for %s epochs. So Stop" % self.early_stop)
                break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)


class PairedTrainer(BasePairedTrainer):

    def _train_epoch(self, **kwargs):
        target_recon_loss, source_recon_loss, dis_loss_all, count = 0.0, 0.0, 0.0, 0.0
        for target_batch, source_batch, label_batch in zip(self.target_batches, self.source_batches, self.label_batches):
            target_batch, source_batch = target_batch.to(self.device), source_batch.to(self.device)
            if not self.add_cons:
                target_code, target_loss = model_helper.run_batch(self.target_model, target_batch, self.model_type)
                self._backward(self.target_optimizer, target_loss)
                source_code, source_loss = model_helper.run_batch(self.source_model, source_batch, self.model_type)
                self._backward(self.source_optimizer, source_loss)
                dis_loss = loss_helper.cal_dis_err(target_code, source_code, label_batch, criterion=self.criterion)
            else:
                # suppose target model is equal to source model
                target_code, target_loss = model_helper.run_batch(self.target_model, target_batch, self.model_type)
                source_code, source_loss = model_helper.run_batch(self.source_model, source_batch, self.model_type)
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


if __name__ == "__main__":
    # test code here, using ae model
    model_type, criterion = "VanillaVAE", "mmd"
    test_log_file, check_dir = "log/%s_add_dis.txt" % model_type, "checkpoint/%s_add_dis/" % model_type
    base_one_model, base_one_opt = tools.get_model_opt(None, tools.get_default_model_class(model_type))
    # change path to which model you want resume from
    model_path = "checkpoint/%s_add_dis/checkpoint-epoch20.pth" % model_type
    test_trainer = PairedTrainer(base_one_model, base_one_opt, criterion, log_file=test_log_file, saved_dir=check_dir,
                                 val_num=100, test_num=100)
    test_trainer.train()
