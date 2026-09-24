"""Training loop and activation diagnostics for Baseline-STEP."""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from net import classifier
import torchlight


def weights_init(module):
    """Use fan-aware convolution initialization and identity BatchNorm."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Processor(object):
    """Train/evaluate the real-only ST-GCN classifier."""

    def __init__(
        self,
        args,
        data_loader,
        in_channels,
        num_classes,
        graph_dict,
        device="cuda:0",
        verbose=True,
    ):
        self.args = args
        self.data_loader = data_loader
        self.num_classes = int(num_classes)
        self.device = device
        self.verbose = verbose
        self.result = {}
        self.meta_info = {"epoch": 0, "iter": 0}
        self.best_epoch = None
        self.best_val_accuracy = -1.0
        self.best_val_loss = float("inf")
        self.best_checkpoint = None
        self._debug_printed = False

        os.makedirs(self.args.work_dir, exist_ok=True)
        self.io = torchlight.IO(
            self.args.work_dir,
            save_log=self.args.save_log,
            print_log=self.args.print_log,
        )
        self.model = classifier.Classifier(
            in_channels,
            self.num_classes,
            graph_dict,
            temporal_kernel_size=9,
        ).to(self.device)
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

        if self.args.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                betas=(0.9, 0.999),
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=self.args.momentum,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError("Unsupported optimizer: {}".format(self.args.optimizer))

        configured_steps = getattr(self.args, "lr_steps", None)
        if configured_steps:
            self.step_epochs = list(configured_steps)
        elif self.args.num_epoch == 500:
            self.step_epochs = [250, 375, 438]
        else:
            self.step_epochs = [
                int(math.ceil(self.args.num_epoch * 0.50)),
                int(math.ceil(self.args.num_epoch * 0.75)),
                int(math.ceil(self.args.num_epoch * 0.876)),
            ]
        self.lr = float(self.args.base_lr)

    def adjust_lr(self):
        epoch = int(self.meta_info["epoch"])
        num_decays = sum(epoch >= step for step in self.step_epochs)
        self.lr = float(self.args.base_lr) * (0.1 ** num_decays)
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    @staticmethod
    def top1_accuracy(logits, labels):
        predictions = np.argmax(logits, axis=1)
        return 100.0 * float(np.mean(predictions == labels))

    @staticmethod
    def _tensor_stats(name, tensor):
        values = tensor.detach().float()
        print(
            "{}: shape={} min={:.8f} max={:.8f} mean={:.8f} std={:.8f} "
            "zero_fraction={:.6f} finite={}".format(
                name,
                tuple(values.shape),
                values.min().item(),
                values.max().item(),
                values.mean().item(),
                values.std(unbiased=False).item(),
                (values == 0).float().mean().item(),
                bool(torch.isfinite(values).all().item()),
            )
        )

    def _register_activation_hooks(self):
        handles = []
        for index, block in enumerate(self.model.st_gcn_networks):
            def report_activation(_module, _inputs, output, block_index=index):
                activations = output[0] if isinstance(output, tuple) else output
                self._tensor_stats(
                    "ST-GCN block {} output".format(block_index + 1),
                    activations,
                )
            handles.append(block.register_forward_hook(report_activation))
        return handles

    def _report_gradients(self):
        print("\n[DEBUG] FIRST-BATCH GRADIENT SUMMARY")
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            if any(token in name for token in (
                "data_bn", "st_gcn_networks.0", "st_gcn_networks.1",
                "st_gcn_networks.2", "fcn",
            )):
                print(
                    "  {}: mean_abs={:.8e} max_abs={:.8e} finite={}".format(
                        name,
                        grad.abs().mean().item(),
                        grad.abs().max().item(),
                        bool(torch.isfinite(grad).all().item()),
                    )
                )

    def per_train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader["train"]
        loss_values = []
        logits_all = []
        labels_all = []

        for batch_index, (data, labels) in enumerate(loader):
            data = data.float().to(self.device)
            labels = labels.long().to(self.device)
            if not torch.isfinite(data).all():
                raise FloatingPointError("Training input contains NaN or Inf.")

            debug_batch = (
                self.meta_info["epoch"] == 0
                and batch_index == 0
                and not self._debug_printed
            )
            hook_handles = self._register_activation_hooks() if debug_batch else []
            self.optimizer.zero_grad()
            logits, features = self.model(data)
            for handle in hook_handles:
                handle.remove()
            loss = self.loss(logits, labels)

            if debug_batch:
                print("\n" + "=" * 72)
                print("FIRST-BATCH BASELINE-STEP DIAGNOSTIC")
                print("=" * 72)
                print("labels:", labels.detach().cpu().numpy())
                self._tensor_stats("input", data)
                self._tensor_stats("logits", logits)
                self._tensor_stats("pooled feature", features)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss at epoch {}, batch {}: {}".format(
                        self.meta_info["epoch"], batch_index, loss.item()
                    )
                )
            loss.backward()

            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(
                        "Non-finite gradient at epoch {}, batch {}, parameter {}."
                        .format(self.meta_info["epoch"], batch_index, name)
                    )
            if debug_batch:
                self._report_gradients()

            grad_clip = float(getattr(self.args, "grad_clip", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            loss_values.append(float(loss.item()))
            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())
            self.meta_info["iter"] += 1

            if debug_batch:
                with torch.no_grad():
                    updated_logits, updated_features = self.model(data)
                self._tensor_stats("logits after first update", updated_logits)
                self._tensor_stats("feature after first update", updated_features)
                print("=" * 72 + "\n")
                self._debug_printed = True

            log_interval = max(1, int(self.args.log_interval))
            if self.meta_info["iter"] % log_interval == 0 and self.verbose:
                self.io.print_log(
                    "\tIter {} | loss {:.6f} | lr {:.6g}".format(
                        self.meta_info["iter"], loss.item(), self.lr
                    )
                )

        if not loss_values:
            raise ValueError("Training loader produced no batches.")
        logits_all = np.concatenate(logits_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        return (
            float(np.mean(loss_values)),
            self.top1_accuracy(logits_all, labels_all),
        )

    def evaluate_loader(self, split):
        if split not in self.data_loader:
            raise KeyError("Missing DataLoader split: {}".format(split))
        self.model.eval()
        losses = []
        logits_fragments = []
        label_fragments = []

        with torch.no_grad():
            for data, labels in self.data_loader[split]:
                data = data.float().to(self.device)
                labels = labels.long().to(self.device)
                logits, _ = self.model(data)
                loss = self.loss(logits, labels)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite {} output/loss.".format(split)
                    )
                losses.append(float(loss.item()))
                logits_fragments.append(logits.cpu().numpy())
                label_fragments.append(labels.cpu().numpy())

        if not losses:
            raise ValueError("{} loader produced no batches.".format(split))
        logits = np.concatenate(logits_fragments, axis=0)
        labels = np.concatenate(label_fragments, axis=0)
        return {
            "loss": float(np.mean(losses)),
            "accuracy": self.top1_accuracy(logits, labels),
            "logits": logits,
            "labels": labels,
        }

    def save_best_checkpoint(self, epoch, val_accuracy, val_loss):
        filename = "epoch{}_valacc{:.2f}_model.pth.tar".format(epoch, val_accuracy)
        path = os.path.join(self.args.work_dir, filename)
        torch.save(self.model.state_dict(), path)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.args.work_dir, "best_model.pth.tar"),
        )
        self.best_epoch = int(epoch)
        self.best_val_accuracy = float(val_accuracy)
        self.best_val_loss = float(val_loss)
        self.best_checkpoint = path
        if self.verbose:
            self.io.print_log(
                "\tNew best checkpoint: epoch {} | val Top1 {:.2f}% | val loss {:.6f}"
                .format(epoch, val_accuracy, val_loss)
            )

    def load_best_model(self):
        path = self.best_checkpoint or os.path.join(
            self.args.work_dir, "best_model.pth.tar"
        )
        if not os.path.isfile(path):
            raise FileNotFoundError("Best checkpoint not found: {}".format(path))
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return path

    def train_tiny_overfit(self, target_accuracy=100.0, max_epochs=1000):
        """Train on the same balanced 16 samples until they are memorized."""
        if "train" not in self.data_loader:
            raise KeyError("Tiny-overfit loader needs a 'train' split.")
        best_accuracy = -1.0
        best_loss = float("inf")
        best_state = None
        best_epoch = -1

        for epoch in range(int(max_epochs)):
            self.meta_info["epoch"] = epoch
            train_loss, online_accuracy = self.per_train()
            metrics = self.evaluate_loader("train")
            accuracy = metrics["accuracy"]
            if (accuracy > best_accuracy) or (
                accuracy == best_accuracy and metrics["loss"] < best_loss
            ):
                best_accuracy = accuracy
                best_loss = metrics["loss"]
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }

            if self.verbose:
                self.io.print_log(
                    "Tiny epoch {:4d} | loss {:.6f} | online acc {:.2f}% | "
                    "eval train acc {:.2f}% | lr {:.6g}".format(
                        epoch, train_loss, online_accuracy, accuracy, self.lr
                    )
                )
            if accuracy >= float(target_accuracy):
                break

        if best_state is None:
            raise RuntimeError("Tiny-overfit stage did not complete an epoch.")
        self.model.load_state_dict(best_state)
        checkpoint_path = os.path.join(self.args.work_dir, "tiny_overfit_model.pth.tar")
        torch.save(self.model.state_dict(), checkpoint_path)
        passed = best_accuracy >= float(target_accuracy)
        if self.verbose:
            self.io.print_log(
                "Tiny-overfit result: {:.2f}% train accuracy at epoch {} ({})"
                .format(best_accuracy, best_epoch, "PASS" if passed else "FAIL")
            )
        return {
            "train_accuracy": best_accuracy,
            "train_loss": best_loss,
            "best_epoch": best_epoch,
            "epochs_run": epoch + 1,
            "passed": passed,
            "checkpoint": checkpoint_path,
        }

    def train(self):
        for split in ("train", "val", "test"):
            if split not in self.data_loader:
                raise KeyError("Missing DataLoader split: {}".format(split))

        last_train_accuracy = 0.0
        for epoch in range(int(self.args.num_epoch)):
            self.meta_info["epoch"] = epoch
            if self.verbose:
                self.io.print_log("Training epoch: {}".format(epoch))
            train_loss, last_train_accuracy = self.per_train()
            if self.verbose:
                self.io.print_log("\ttrain_loss: {:.6f}".format(train_loss))
                self.io.print_log("\ttrain Top1: {:.2f}%".format(last_train_accuracy))
                self.io.print_log("\tlr: {:.6g}".format(self.lr))

            if epoch % int(self.args.eval_interval) == 0 or epoch + 1 == int(self.args.num_epoch):
                val = self.evaluate_loader("val")
                if self.verbose:
                    self.io.print_log("Validation epoch: {}".format(epoch))
                    self.io.print_log("\tval_loss: {:.6f}".format(val["loss"]))
                    self.io.print_log("\tval Top1: {:.2f}%".format(val["accuracy"]))
                is_better = val["accuracy"] > self.best_val_accuracy or (
                    val["accuracy"] == self.best_val_accuracy
                    and val["loss"] < self.best_val_loss
                )
                if is_better:
                    self.save_best_checkpoint(epoch, val["accuracy"], val["loss"])

        if self.best_checkpoint is None:
            raise RuntimeError("No validation checkpoint was produced.")
        self.load_best_model()
        train_metrics = self.evaluate_loader("train")
        test_metrics = self.evaluate_loader("test")
        self.result = test_metrics["logits"]
        self.label = test_metrics["labels"]

        if self.verbose:
            self.io.print_log("Final evaluation using best validation checkpoint:")
            self.io.print_log("\tBest epoch: {}".format(self.best_epoch))
            self.io.print_log("\tTrain Top1: {:.2f}%".format(train_metrics["accuracy"]))
            self.io.print_log("\tBest validation Top1: {:.2f}%".format(self.best_val_accuracy))
            self.io.print_log("\tTest loss: {:.6f}".format(test_metrics["loss"]))
            self.io.print_log("\tTest Top1: {:.2f}%".format(test_metrics["accuracy"]))

        return {
            "best_epoch": self.best_epoch,
            "train_accuracy": train_metrics["accuracy"],
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_loss": self.best_val_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
        }

    def generate_predictions(self, data, load_best=True):
        if load_best:
            self.load_best_model()
        dataset = self.data_loader.get("prediction")
        if dataset is None:
            from utils.loader import TrainTestLoader
            dataset = TrainTestLoader(data, np.zeros(len(data), dtype=np.int64))
        self.model.eval()
        logits = []
        with torch.no_grad():
            for each_data, _ in torch.utils.data.DataLoader(
                dataset, batch_size=32, shuffle=False
            ):
                output, _ = self.model(each_data.float().to(self.device))
                logits.append(output.cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        return np.argmax(logits, axis=1), logits
