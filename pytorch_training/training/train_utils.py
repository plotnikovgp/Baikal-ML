from pathlib import Path
from tqdm import tqdm
import wandb
import torch
import torch_geometric
import json
import numpy as np
from metrics import binary_clf_metrics


def _run_model(
    model,
    data,
    is_graph=False,
    is_classification=False,
    is_track_cascade_tres_train=False,
    is_angle_reconstruction=False,
    is_angle_and_track_cascade=False,
    is_angle_reconstruction_sigma_tune=False,
    is_direction=False,
    is_domain_adaptation=False,
    track_cascade_model=None,
    dataset_idx=None,
    **kwargs,
):
    # print args
    if is_graph:
        data = data.to(model.device)
        data.edge_index = torch_geometric.utils.sort_edge_index(data.edge_index)
        y_true = data.y
        output = model(data.x, data.edge_index, data.batch).squeeze()
        if is_angle_reconstruction:
            y_true = y_true.reshape(output.shape)
            output = output / output.norm(dim=1, keepdim=True)
            y_pred = output
    else:
        x, y_true, mask = data

        x, y_true, mask = (
            x.to(model.device),
            y_true.to(model.device),
            mask.to(model.device),
        )

        if track_cascade_model:
            with torch.no_grad():
                track_cascade_prob = track_cascade_model(x, mask)[:, :, 0]
            x = torch.cat([x, track_cascade_prob.unsqueeze(-1)], dim=-1)

        if is_domain_adaptation:
            output, domain_output, _ = model(x, mask)

            batch_size = x.shape[0]
            domain_true = torch.full((batch_size,), dataset_idx, dtype=torch.long, device=model.device)
            domain_pred = domain_output
        else:
            output = model(x, mask)
            domain_pred = None
            domain_true = None

        if is_track_cascade_tres_train:
            y_true = y_true.reshape(-1, y_true.shape[-1])
            output = output.reshape(-1, output.shape[-1]).squeeze()
        elif is_angle_reconstruction or is_domain_adaptation:
            output = output / output.norm(dim=1, keepdim=True)
        elif is_angle_reconstruction_sigma_tune:
            pass
        elif is_direction:
            norms = output[:, :3].norm(dim=-1, keepdim=True)
            normalized_values = output[:, :3] / norms
            output = torch.cat((normalized_values, output[:, 3:]), dim=-1)
        else:
            y_true = y_true.reshape(-1)
            output = output.reshape(-1, output.shape[-1]).squeeze()
        if not (is_angle_reconstruction or is_direction or is_angle_reconstruction_sigma_tune or is_domain_adaptation):
            mask = mask.reshape(-1)
            y_true = y_true[mask != 0]
            output = output[mask != 0]

    if is_classification:
        y_pred = torch.sigmoid(output[:, 1])
    elif is_track_cascade_tres_train:
        y_pred = torch.zeros_like(y_true)
        y_pred[:, 0] = torch.sigmoid(output[:, 1])
        y_pred[:, 1] = output[:, -1]
    elif is_angle_and_track_cascade:
        y_pred = torch.zeros_like(y_true)
        y_pred[:, 1] = output[:, -2]
        y_pred[:, 2] = output[:, -1]
        y_pred[:, 0] = torch.sigmoid(output[:, 1])
    else:
        y_pred = output

    if is_domain_adaptation:
        return output, y_pred, y_true, domain_pred, domain_true
    else:
        return output, y_pred, y_true


def train_iters(
    model,
    optimizer,
    train_loader,
    criterion,
    metrics_calc_fun,
    warmup_scheduler=None,
    warmup_steps=0,
    scheduler=None,
    accumulate_grad_steps=1,
    num_iters=1,
    grad_clip_value=None,
    is_domain_adaptation=False,
    **kwargs,
):
    model.train()
    y_pred_hist = None
    y_true_hist = None
    domain_pred_hist = None
    domain_true_hist = None
    loss_hist = []
    domain_loss_hist = []
    total_loss_hist = []
    loss_accum = 0.0

    for iter in range(num_iters):
        data = next(train_loader)

        dataset_idx = data[2] if isinstance(data, tuple) and len(data) > 2 and is_domain_adaptation else None
        if isinstance(data, tuple) and len(data) > 2 and is_domain_adaptation:
            data = data[:2] + data[3:]

        if is_domain_adaptation:
            output, y_pred, y_true, domain_pred, domain_true = _run_model(
                model,
                data,
                is_domain_adaptation=True,
                dataset_idx=dataset_idx,
                **kwargs,
            )
            loss = criterion(y_pred, y_true, domain_pred, domain_true)

            domain_pred_hist = (
                torch.cat((domain_pred_hist, domain_pred.detach()), dim=0)
                if domain_pred_hist is not None
                else domain_pred.detach()
            )
            domain_true_hist = (
                torch.cat((domain_true_hist, domain_true.detach()), dim=0)
                if domain_true_hist is not None
                else domain_true.detach()
            )
        else:
            output, y_pred, y_true = _run_model(
                model,
                data,
                **kwargs,
            )
            loss = criterion(output, y_true)

        loss.backward()
        loss_accum += loss.item()
        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        if iter % accumulate_grad_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    if warmup_scheduler.last_step + 1 >= warmup_steps:
                        scheduler.step(loss_accum)
                        loss_accum = 0.0

        loss_hist.append(loss.item())

        y_pred_hist = torch.cat((y_pred_hist, y_pred), dim=0) if y_pred_hist is not None else y_pred

        y_true_hist = torch.cat((y_true_hist, y_true), dim=0) if y_true_hist is not None else y_true

    train_metrics = metrics_calc_fun(y_pred_hist.detach().cpu(), y_true_hist.detach().cpu())
    train_metrics["loss"] = sum(loss_hist) / len(loss_hist) if loss_hist else None
    train_metrics["lr"] = optimizer.param_groups[0]["lr"]

    if is_domain_adaptation:
        domain_preds = domain_pred_hist.argmax(dim=1)
        domain_labels = domain_true_hist
        domain_accuracy = (domain_preds == domain_labels).float().mean().item()
        train_metrics["domain_accuracy"] = domain_accuracy
        domain_bin_metrics = binary_clf_metrics(domain_preds.cpu().numpy(), domain_labels.cpu().numpy())
        train_metrics.update({"domain_" + k: v for k, v in domain_bin_metrics.items()})
    return train_metrics


def validate_single(
    model,
    val_loader,
    criterion,
    metrics_calc_fun,
    return_preds=False,
    is_domain_adaptation=False,
    dataset_idx=0,
    **kwargs,
) -> dict[str, float]:
    y_pred_hist = None
    y_true_hist = None
    domain_pred_hist = None
    domain_true_hist = None
    loss_hist = []
    domain_loss_hist = []

    model.train()
    with torch.no_grad():
        for data in val_loader:
            if is_domain_adaptation:
                output, y_pred, y_true, domain_pred, domain_true = _run_model(
                    model,
                    data,
                    is_domain_adaptation=True,
                    dataset_idx=dataset_idx,
                    **kwargs,
                )
                loss = criterion(y_pred, y_true, domain_pred, domain_true)

                domain_pred_hist = (
                    torch.cat((domain_pred_hist, domain_pred.detach().cpu()), dim=0)
                    if domain_pred_hist is not None
                    else domain_pred.detach().cpu()
                )
                domain_true_hist = (
                    torch.cat((domain_true_hist, domain_true.detach().cpu()), dim=0)
                    if domain_true_hist is not None
                    else domain_true.detach().cpu()
                )
            else:
                output, y_pred, y_true = _run_model(
                    model,
                    data,
                    **kwargs,
                )
                loss = criterion(output, y_true)

            loss_hist.append(loss.item())

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

    if return_preds:
        return y_pred_hist, y_true_hist
    val_metrics = metrics_calc_fun(y_pred_hist, y_true_hist)
    val_metrics["loss"] = sum(loss_hist) / len(loss_hist) if loss_hist else None

    if is_domain_adaptation and domain_pred_hist is not None:
        domain_preds = domain_pred_hist.argmax(dim=1)
        domain_labels = domain_true_hist
        domain_accuracy = (domain_preds == domain_labels).float().mean().item()
        val_metrics["domain_accuracy"] = domain_accuracy
        domain_bin_metrics = binary_clf_metrics(domain_preds, domain_labels)
        domain_bin_metrics = {"domain_" + k: v for k, v in domain_bin_metrics.items()}
        val_metrics.update(domain_bin_metrics)
    return val_metrics


def validate(
    model,
    val_loader,
    criterion,
    metrics_calc_fun,
    return_preds=False,
    dataset_names=None,
    **kwargs,
) -> dict[str, float]:
    if not isinstance(val_loader, list):
        return validate_single(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            metrics_calc_fun=metrics_calc_fun,
            return_preds=return_preds,
            **kwargs,
        )

    all_metrics = {}

    for i, loader in enumerate(val_loader):
        dataset_metrics = validate_single(
            model=model,
            val_loader=loader,
            criterion=criterion,
            metrics_calc_fun=metrics_calc_fun,
            return_preds=False,
            dataset_idx=i,
            **kwargs,
        )

        dataset_prefix = dataset_names[i]
        for k, v in dataset_metrics.items():
            if isinstance(v, (int, float)):
                all_metrics[f"{dataset_prefix}_{k}"] = v

    return all_metrics


def train(
    model,
    train_fun,
    validate_fun,
    train_params_str,
    train_fun_kwargs={},
    validate_fun_kwargs={},
    log_every=1,
    val_every_epochs=1,
    epochs=20000,
    use_wandb=False,
    save_best_model=True,
    valid_main_metric="loss",
    model_save_dir="models",
    validate_before_train=True,
    save_best_per_dataset=False,
    dataset_names=None,
):
    best_val_metrics = {}
    validate_before_train = True
    train_logs_ = {}
    iters_current = 0
    iters_per_epoch = len(train_fun_kwargs["dataset"]) / train_fun_kwargs["num_iters"]
    total_iters = int(epochs * iters_per_epoch)
    print("Num steps in one epoch: ", iters_per_epoch)
    cur_epoch = 0

    if dataset_names is None and isinstance(validate_fun_kwargs.get("val_loader", None), list):
        num_datasets = len(validate_fun_kwargs["val_loader"])
        dataset_names = [f"dataset_{i}" for i in range(num_datasets)]

    with tqdm(range(total_iters), unit="batch", dynamic_ncols=True) as step_iter:
        for _ in step_iter:
            if not validate_before_train:
                train_logs = train_fun(model, **train_fun_kwargs)
                train_logs_ = {"train/" + k: train_logs[k] for k in sorted(list(train_logs.keys()))}
                iters_current += train_fun_kwargs["num_iters"]
                if iters_current >= iters_per_epoch:
                    cur_epoch += iters_current // iters_per_epoch
                    iters_current = iters_current % iters_per_epoch

                train_logs_["train/epoch"] = cur_epoch
                if use_wandb:
                    wandb.log(train_logs_)
                to_print = {k: train_logs_[k] for k in ["train/loss", "train/epoch"]}

                step_iter.set_description(str(to_print))

            if validate_before_train or cur_epoch % val_every_epochs == 0:
                validate_before_train = False
                val_logs = validate_fun(model, **validate_fun_kwargs)

                val_logs_ = {"val/" + k: v for k, v in val_logs.items()}
                if use_wandb:
                    wandb.log(val_logs_)

                if save_best_model or save_best_per_dataset:
                    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

                    for dataset_name in dataset_names or [None]:
                        if dataset_name is None:
                            if valid_main_metric in val_logs:
                                metric_key = valid_main_metric
                                metric_value = val_logs[metric_key]
                                dataset_label = ""

                                if (
                                    dataset_label not in best_val_metrics
                                    or metric_value < best_val_metrics[dataset_label]
                                ):
                                    best_val_metrics[dataset_label] = metric_value

                                    save_path = f"{model_save_dir}/best{dataset_label}.ckpt"
                                    torch.save(model.state_dict(), save_path)

                                    with open(
                                        f"{model_save_dir}/best_val_metrics{dataset_label}.txt",
                                        "w",
                                    ) as f:
                                        f.write(json.dumps(val_logs))
                                        f.write("\n" + json.dumps(train_logs_))

                                    with open(
                                        f"{model_save_dir}/train_config{dataset_label}.yaml",
                                        "w",
                                    ) as f:
                                        f.write(train_params_str)
                        else:
                            metric_key = f"{dataset_name}_{valid_main_metric}"
                            if metric_key in val_logs:
                                metric_value = val_logs[metric_key]
                                dataset_label = f"_{dataset_name}"

                                if (
                                    dataset_label not in best_val_metrics
                                    or metric_value < best_val_metrics[dataset_label]
                                ):
                                    best_val_metrics[dataset_label] = metric_value

                                    if save_best_per_dataset or (save_best_model and dataset_name == dataset_names[0]):
                                        save_path = f"{model_save_dir}/best{dataset_label}.ckpt"
                                        torch.save(model.state_dict(), save_path)

                                        with open(
                                            f"{model_save_dir}/best_val_metrics{dataset_label}.txt",
                                            "w",
                                        ) as f:
                                            f.write(json.dumps(val_logs))
                                            f.write("\n" + json.dumps(train_logs_))
