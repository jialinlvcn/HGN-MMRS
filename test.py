import argparse
import os
import enlighten
import time
import torch
from torch.utils.data import DataLoader

import mmrs_utils
from task_config import get_dataset, get_model, Augsburg_cmap, Houston2013_cmap

parser = argparse.ArgumentParser(description="Multi Model Testing")
parser.add_argument("--data_set", default="Houston2013", type=str)
parser.add_argument("--data_dir", default="./data", type=str)
parser.add_argument("--log_save_dir", default="./img", type=str)
parser.add_argument("--model_save_dir", default="./checkpoints", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--patch_size", default=7, type=int)
parser.add_argument("--model", default="HGN", type=str)
args = parser.parse_args()


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
    return acc_meter.avg, acc_meter.cale_kappa()


def main():
    if args.data_set == "Trento":
        p_list = [0.05, 0.1, 0.2, 1]
        cmap = mmrs_utils.generate_cmap(7)
    elif args.data_set == "Houston2013":
        p_list = [0.1, 0.2, 0.5, 1]
        cmap = Houston2013_cmap
    elif args.data_set == "Augsburg":
        p_list = [0.1, 0.2, 0.5, 1]
        cmap = Augsburg_cmap

    manager = enlighten.get_manager()
    status_bar = manager.status_bar("", color="white_on_green", justify=enlighten.Justify.CENTER)
    _, _, data = get_dataset(args.data_dir, args.data_set, args.patch_size, 1.0)
    _ = mmrs_utils.hsi2rgb(
        data.get("HSI"),
        [59, 40, 23],
        save_path=f"./img/{args.data_set}_HSI.png",
    )

    for p in p_list:
        time.sleep(0.1)
        message = f"Evaluating the dataset {args.data_set} on the {args.model} model \
            and the training sampling rate is {p * 100}%"
        status_bar.update(message)
        model = get_model(args.model, args.data_set, args.patch_size)
        save_name = os.path.join(args.model_save_dir, f"{args.data_set}-{args.model}-{int(p * 100)}.pt")

        model_dict = torch.load(save_name, weights_only=True)
        model.load_state_dict(model_dict)

        _, test_set, _ = get_dataset(args.data_dir, args.data_set, args.patch_size, p)
        _, _, data = get_dataset(args.data_dir, args.data_set, args.patch_size, 1, ignore=[0])
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        testacc, test_kappa = evalution(model, test_loader, args.device, manager)
        print(f"{int(p * 100)}% dataset test OA = {testacc:.2f} %  kappa = {test_kappa:.4f}")
        model.eval()
        mmrs_utils.draw_result(
            data,
            f"{args.log_save_dir}/{args.data_set}-{args.model}-{int(p * 100)}-result.png",
            model=model,
            manager=manager,
            device=args.device,
            patch_size=args.patch_size,
            draw_gt=False,
            ignore=[],
            batch_size=args.batch_size * 8,
            cmap=cmap,
        )
    manager.stop()

if __name__ == "__main__":
    main()
