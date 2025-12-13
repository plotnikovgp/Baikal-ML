import json
from pathlib import Path

import torch
from tqdm import tqdm

import wandb
from metrics import BaseMetrics, BinaryClassificationMetrics


class Trainer:
    def __init__(
        self,
        model,
        train_type,
        optimizer,
        warmup_scheduler=None,
        warmup_steps=0,
        scheduler=None,
        accumulate_grad_steps=1,
        grad_clip_value=None,
        use_wandb=False,
        tensorboard_writer=None,
        model_save_dir="models",
        valid_main_metric="loss",
    ):
        self.model = model
        self.train_type = train_type
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.warmup_steps = warmup_steps
        self.scheduler = scheduler
        self.accumulate_grad_steps = accumulate_grad_steps
        self.grad_clip_value = grad_clip_value
        self.use_wandb = use_wandb
        self.tensorboard_writer = tensorboard_writer
        self.model_save_dir = model_save_dir
        self.valid_main_metric = valid_main_metric

        self.criterion = train_type.get_criterion()
        self.metrics_fn = train_type.get_metrics_function()
        self.domain_metrics_fn = BinaryClassificationMetrics(min_recall=0.8)

        metrics_to_maximize = ["auc", "precision", "recall", "accuracy", "domain_accuracy"]
        self.maximize_metric = any(m in valid_main_metric for m in metrics_to_maximize)

    def train_iters(self, train_loader, num_iters=1, min_recall=None):
        self.model.train()
        y_pred_hist = None
        y_true_hist = None
        domain_pred_hist = None
        domain_true_hist = None
        loss_hist = {}
        loss_accum = 0.0
        is_domain_adaptation = self.train_type.get_train_kwargs().get("is_domain_adaptation", False)

        for iter_idx in range(num_iters):
            data = next(train_loader)

            result = self.train_type.process_batch(self.model, data)
            output, y_pred, y_true = result["output"], result["y_pred"], result["y_true"]
            domain_pred = result.get("domain_pred")
            domain_true = result.get("domain_true")

            if is_domain_adaptation:
                loss = self.criterion(output, y_true, domain_pred, domain_true)
                domain_true_event = (
                    domain_true[0] if isinstance(domain_true, tuple) else domain_true
                )
                domain_pred_hist = (
                    torch.cat((domain_pred_hist, domain_pred.detach()), dim=0)
                    if domain_pred_hist is not None
                    else domain_pred.detach()
                )
                domain_true_hist = (
                    torch.cat((domain_true_hist, domain_true_event.detach()), dim=0)
                    if domain_true_hist is not None
                    else domain_true_event.detach()
                )
            else:
                loss = self.criterion(output, y_true)
                domain_true_event = None

            if not isinstance(loss, dict):
                loss = {"loss": loss}

            for k, v in loss.items():
                if k not in loss_hist:
                    loss_hist[k] = []
                loss_hist[k].append(v.item())

            loss_to_backward = loss["loss"]
            loss_to_backward.backward()
            loss_accum += loss_to_backward.item()

            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

            if iter_idx % self.accumulate_grad_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.warmup_scheduler is not None:
                    with self.warmup_scheduler.dampening():
                        if self.warmup_scheduler.last_step + 1 >= self.warmup_steps:
                            if isinstance(
                                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                            ):
                                self.scheduler.step(loss_accum)
                                loss_accum = 0.0
                            elif self.scheduler is not None:
                                self.scheduler.step()
                elif self.scheduler is not None and not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()

            is_unlabeled_batch = (
                is_domain_adaptation
                and domain_true_event is not None
                and domain_true_event[0].item() != 0
            )

            if not is_unlabeled_batch:
                y_pred_hist = (
                    torch.cat((y_pred_hist, y_pred), dim=0) if y_pred_hist is not None else y_pred
                )
                y_true_hist = (
                    torch.cat((y_true_hist, y_true), dim=0) if y_true_hist is not None else y_true
                )

        if y_pred_hist is not None and y_true_hist is not None:
            if min_recall is not None:
                train_metrics = self.metrics_fn(
                    y_pred_hist.detach().cpu(), y_true_hist.detach().cpu(), min_recall=min_recall
                )
            else:
                train_metrics = self.metrics_fn(
                    y_pred_hist.detach().cpu(), y_true_hist.detach().cpu()
                )
        else:
            train_metrics = {}

        for k, v in loss_hist.items():
            train_metrics[k] = sum(v) / len(v) if v else None
        train_metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        if is_domain_adaptation:
            domain_preds = domain_pred_hist.argmax(dim=1)
            domain_labels = domain_true_hist
            domain_accuracy = (domain_preds == domain_labels).float().mean().item()
            train_metrics["domain_accuracy"] = domain_accuracy
            domain_bin_metrics = self.domain_metrics_fn(
                domain_preds.cpu().numpy(), domain_labels.cpu().numpy()
            )
            train_metrics.update({"domain_" + k: v for k, v in domain_bin_metrics.items()})

        return train_metrics

    def validate_single(self, val_loader, dataset_idx=0, min_recall=None, val_mode=False):
        y_pred_hist = None
        y_true_hist = None
        domain_pred_hist = None
        domain_true_hist = None
        loss_hist = {}
        is_domain_adaptation = self.train_type.get_train_kwargs().get("is_domain_adaptation", False)

        self.model.eval()
        with torch.no_grad():
            for data in val_loader:
                result = self.train_type.process_batch(self.model, data, dataset_idx=dataset_idx)
                output, y_pred, y_true = result["output"], result["y_pred"], result["y_true"]
                domain_pred = result.get("domain_pred")
                domain_true = result.get("domain_true")

                if is_domain_adaptation:
                    domain_true_event = (
                        domain_true[0] if isinstance(domain_true, tuple) else domain_true
                    )
                    domain_pred_hist = (
                        torch.cat((domain_pred_hist, domain_pred.detach()), dim=0)
                        if domain_pred_hist is not None
                        else domain_pred.detach()
                    )
                    domain_true_hist = (
                        torch.cat((domain_true_hist, domain_true_event.detach()), dim=0)
                        if domain_true_hist is not None
                        else domain_true_event.detach()
                    )
                    if (
                        len(y_true.shape) == 1
                        and len(output.shape) > 1
                        and y_true.shape[0] == output.shape[0]
                    ):
                        loss = {"loss": torch.tensor(0.0, device=output.device)}
                    else:
                        loss = self.criterion(output, y_true, None, None)
                else:
                    loss = self.criterion(output, y_true)

                if not isinstance(loss, dict):
                    loss = {"loss": loss}

                for k, v in loss.items():
                    if k not in loss_hist:
                        loss_hist[k] = []
                    loss_hist[k].append(v.item())

                y_pred_hist = (
                    torch.cat((y_pred_hist, y_pred.detach().cpu()), dim=0)
                    if y_pred_hist is not None
                    else y_pred.detach().cpu()
                )
                y_true_hist = (
                    torch.cat((y_true_hist, y_true.detach().cpu()), dim=0)
                    if y_true_hist is not None
                    else y_true.detach().cpu()
                )

        has_real_labels = not (len(y_true_hist.shape) == 1 and torch.all(y_true_hist == 0))

        if not has_real_labels:
            val_metrics = {}
        elif min_recall is not None:
            val_metrics = self.metrics_fn(y_pred_hist, y_true_hist, min_recall=min_recall)
        else:
            val_metrics = self.metrics_fn(y_pred_hist, y_true_hist)

        if is_domain_adaptation and not val_mode:
            domain_preds = domain_pred_hist.argmax(dim=1)
            domain_labels = domain_true_hist
            domain_accuracy = (domain_preds == domain_labels).float().mean().item()
            val_metrics["domain_accuracy"] = domain_accuracy
            domain_bin_metrics = self.domain_metrics_fn(
                domain_preds.cpu().numpy(), domain_labels.cpu().numpy()
            )
            val_metrics.update({"domain_" + k: v for k, v in domain_bin_metrics.items()})

        for k, v in loss_hist.items():
            val_metrics[k] = sum(v) / len(v) if v else None

        return val_metrics

    def validate(
        self, val_loader, dataset_names=None, min_recall=None, val_mode=False, global_step=0
    ):
        if not isinstance(val_loader, list):
            return self.validate_single(val_loader, min_recall=min_recall, val_mode=val_mode)

        all_metrics = {}

        for i, loader in enumerate(val_loader):
            if isinstance(self.metrics_fn, BaseMetrics):
                self.metrics_fn.set_dataset_name(dataset_names[i])

            dataset_metrics = self.validate_single(
                loader, dataset_idx=i, min_recall=min_recall, val_mode=val_mode
            )

            if val_mode:
                print(f"Dataset {dataset_names[i]}")
                for key, value in dataset_metrics.items():
                    print(f"{key}: {value}")
                print("\n")

            dataset_prefix = dataset_names[i]
            for k, v in dataset_metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[f"{dataset_prefix}_{k}"] = v

        return all_metrics

    def train(
        self,
        train_loader,
        val_loader,
        train_params_str,
        num_iters=1,
        log_every=1,
        val_every_epochs=1,
        epochs=20000,
        save_best_model=True,
        validate_before_train=True,
        save_best_per_dataset=False,
        dataset_names=None,
        val_mode=False,
        min_recall=None,
        train_dataset_size=None,
    ):
        best_val_metrics = {}
        train_logs_ = {}
        iters_current = 0
        total_steps = 0

        if train_dataset_size is not None:
            iters_per_epoch = train_dataset_size / num_iters
        elif hasattr(train_loader, "dataset"):
            iters_per_epoch = len(train_loader.dataset) / num_iters
        else:
            iters_per_epoch = 1000
        total_iters = int(epochs * iters_per_epoch)
        print("Num steps in one epoch: ", iters_per_epoch)
        cur_epoch = 0

        if dataset_names is None and isinstance(val_loader, list):
            num_datasets = len(val_loader)
            dataset_names = [f"dataset_{i}" for i in range(num_datasets)]

        with tqdm(range(total_iters), unit="batch", dynamic_ncols=True) as step_iter:
            for _ in step_iter:
                if not validate_before_train and not val_mode:
                    train_logs = self.train_iters(
                        train_loader, num_iters=num_iters, min_recall=min_recall
                    )
                    train_logs_ = {
                        "train/" + k: float(train_logs[k]) for k in sorted(list(train_logs.keys()))
                    }
                    iters_current += num_iters
                    if iters_current >= iters_per_epoch:
                        cur_epoch += iters_current // iters_per_epoch
                        iters_current = iters_current % iters_per_epoch

                    train_logs_["train/epoch"] = cur_epoch
                    total_steps += 1

                    if self.use_wandb and not self.tensorboard_writer:
                        wandb.log(train_logs_)
                    elif self.tensorboard_writer is not None:
                        for key, value in train_logs_.items():
                            if value is not None and isinstance(value, (int, float)):
                                self.tensorboard_writer.add_scalar(key, value, total_steps)

                    to_print = {
                        k: train_logs_[k] for k in train_logs_ if "loss" in k or k in ["epoch"]
                    }
                    step_iter.set_description(str(to_print))

                if validate_before_train or cur_epoch % val_every_epochs == 0 or val_mode:
                    validate_before_train = False
                    val_logs = self.validate(
                        val_loader,
                        dataset_names=dataset_names,
                        min_recall=min_recall,
                        val_mode=val_mode,
                        global_step=total_steps,
                    )
                    val_logs_ = {"val/" + k: v for k, v in val_logs.items()}

                    if self.use_wandb and not self.tensorboard_writer:
                        wandb.log(val_logs_)
                    elif self.tensorboard_writer is not None:
                        for key, value in val_logs_.items():
                            if value is not None and isinstance(value, (int, float)):
                                self.tensorboard_writer.add_scalar(key, value, total_steps)

                    if save_best_model or save_best_per_dataset:
                        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
                        self._save_best_models(
                            val_logs,
                            dataset_names,
                            best_val_metrics,
                            train_logs_,
                            train_params_str,
                            val_mode,
                            save_best_per_dataset,
                        )

                if val_mode:
                    break

    def _save_best_models(
        self,
        val_logs,
        dataset_names,
        best_val_metrics,
        train_logs_,
        train_params_str,
        val_mode,
        save_best_per_dataset,
    ):
        for dataset_name in dataset_names or [None]:
            if dataset_name is None:
                if self.valid_main_metric in val_logs:
                    metric_key = self.valid_main_metric
                    metric_value = val_logs[metric_key]
                    dataset_label = ""
                    self._check_and_save(
                        metric_key,
                        metric_value,
                        dataset_label,
                        best_val_metrics,
                        val_logs,
                        train_logs_,
                        train_params_str,
                        val_mode,
                    )
            else:
                metric_key = f"{dataset_name}_{self.valid_main_metric}"
                if metric_key in val_logs:
                    metric_value = val_logs[metric_key]
                    dataset_label = f"_{dataset_name}"
                    should_save = save_best_per_dataset or (dataset_name == dataset_names[0])
                    if should_save:
                        self._check_and_save(
                            metric_key,
                            metric_value,
                            dataset_label,
                            best_val_metrics,
                            val_logs,
                            train_logs_,
                            train_params_str,
                            val_mode,
                        )

    def _check_and_save(
        self,
        metric_key,
        metric_value,
        dataset_label,
        best_val_metrics,
        val_logs,
        train_logs_,
        train_params_str,
        val_mode,
    ):
        is_best = False
        if dataset_label not in best_val_metrics:
            is_best = True
        elif self.maximize_metric:
            is_best = metric_value > best_val_metrics[dataset_label]
        else:
            is_best = metric_value < best_val_metrics[dataset_label]

        if is_best and not val_mode:
            best_val_metrics[dataset_label] = metric_value
            print(
                f"[CHECKPOINT] Saving best model: {metric_key}={metric_value:.4f} (maximize={self.maximize_metric})"
            )

            save_path = f"{self.model_save_dir}/best{dataset_label}.ckpt"
            torch.save(self.model.state_dict(), save_path)

            with open(f"{self.model_save_dir}/best_val_metrics{dataset_label}.txt", "w") as f:
                f.write(json.dumps(val_logs, indent=4))
                f.write("\n" + json.dumps(train_logs_, indent=4))

            with open(f"{self.model_save_dir}/train_config{dataset_label}.yaml", "w") as f:
                f.write(train_params_str)
