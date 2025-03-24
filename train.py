import argparse
import os
import random
import time
import numpy as np
import torch
import enlighten
import torch.nn as nn
from torch.utils.data import DataLoader

import mmrs_utils
from task_config import get_dataset, get_model

parser = argparse.ArgumentParser(description="Multi Model Training")
parser.add_argument("--data_set", default="Augsburg", type=str)
parser.add_argument("--data_dir", default="./data", type=str)
parser.add_argument("--seed", default=666, type=int, help="seed")
parser.add_argument("--log_save_dir", default="./img", type=str)
parser.add_argument("--model_save_dir", default="./checkpoints", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--patch_size", default=7, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--model", default="HGN", type=str)
args = parser.parse_args()


def seed_everything(seed=111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(args.seed)
loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)


def train(epoch, model, train_loader, optimizer, device, manager: enlighten.Manager):
    torch.autograd.set_detect_anomaly(True)
    model = model.to(device)
    model.train()
    acc_meter = mmrs_utils.AverageMeter()
    loss_meter = mmrs_utils.LossMeter()

    bar_format = (
        "{desc}{desc_pad}{percentage:3.0f}%|{bar}| "
        + "loss={loss:.2e} "
        + "OA={acc:.2f}% "
        + "[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]"
    )
    ticks = manager.counter(
        total=len(train_loader),
        desc=f"[Epoch {epoch: 3d}] Training",
        unit="pixel",
        color="green",
        bar_format=bar_format,
        leave=False,
    )

    for _, (data, gt, _) in enumerate(train_loader):
        data, gt = [d.to(device) for d in data], gt.to(device)
        if args.model == "End":
            output = model(data)
            loss_ce = loss(output[0], gt)
            loss_mse_list = []
            for re_input, input in zip(output[1], data):
                loss_mse_list.append(mse_loss(re_input, input))
            loss_mse = sum(loss_mse_list)
            lossrt = loss_ce + loss_mse
        else:
            output = model(data)
            lossrt = loss(output, gt)
        loss_meter.update(lossrt, len(gt))
        lossrt.backward()
        optimizer.step()
        optimizer.zero_grad()
        try:
            acc_meter.update(torch.argmax(output, dim=1), gt)
        except TypeError:
            acc_meter.update(torch.argmax(output[0], dim=1), gt)
        ticks.update(loss=loss_meter.avg, acc=acc_meter.avg)
    ticks.close()
    return acc_meter.avg, loss_meter.avg


def evalution(model: torch.nn.Module, test_loader: DataLoader, device: str, manager: enlighten.Manager):
    model.eval()
    model = model.to(device)
    acc_meter = mmrs_utils.AverageMeter()
    ticks = manager.counter(total=len(test_loader), desc="Evalution testset", unit="pixel", color="red", leave=False)
    with torch.no_grad():
        for data, gt, _ in test_loader:
            data, gt = [d.to(device) for d in data], gt.to(device)
            output = model(data)
            try:
                acc_meter.update(torch.argmax(output, dim=1), gt)
            except TypeError:
                acc_meter.update(torch.argmax(output[0], dim=1), gt)
            ticks.update()
    ticks.close()
    return acc_meter.avg


def main():
    if args.data_set == "Trento":
        p_list = [0.05, 0.1, 0.2, 1]
    elif args.data_set == "Houston2013":
        p_list = [0.1, 0.2, 0.5, 1]
    elif args.data_set == "Augsburg":
        p_list = [0.1, 0.2, 0.5, 1]

    manager = enlighten.get_manager()
    status_bar = manager.status_bar("", color="white_on_green", justify=enlighten.Justify.CENTER)
    for p in p_list:
        time.sleep(0.1)
        message = f"Training the dataset {args.data_set} on the {args.model} model \
            and the training sampling rate is {p * 100}%"
        status_bar.update(message)
        train_set, test_set, _ = get_dataset(args.data_dir, args.data_set, args.patch_size, p)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        model = get_model(args.model, args.data_set, args.patch_size)

        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        max_testacc = 0

        save_name = os.path.join(args.model_save_dir, f"{args.data_set}-{args.model}-{int(p * 100)}.pt")
        for i in range(args.epochs):
            train_acc, train_loss = train(i, model, train_loader, optimizer, args.device, manager)
            print(f"[{i:3d}]  train OA = {train_acc:.2f} % | train loss = {train_loss:.5f}")
            if i % 5 == 4:
                testacc = evalution(model, test_loader, args.device, manager)

                if testacc > max_testacc:
                    max_testacc = testacc
                    torch.save(model.to("cpu").state_dict(), save_name)
                print(f"[{i:3d}]  test OA = {testacc:.2f} %")

        del model
    manager.stop()

if __name__ == "__main__":
    main()
