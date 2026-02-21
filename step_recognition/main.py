import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
import os.path as osp
from utils import get_logger, set_seed, create_outdir, build_lr_scheduler
from model import build_model
from datasets import build_data_loader
from criterions import build_criterion
from trainer import build_trainer, build_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/miniroad_thumos_kinetics.yaml"
    )
    parser.add_argument("--eval", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--no_rgb", action="store_true")
    parser.add_argument("--no_flow", action="store_true")
    # Added explicit resume flag if you ever want to force it
    parser.add_argument("--resume", action="store_true", help="Auto-resume from last checkpoint")
    args = parser.parse_args()

    # combine argparse and yaml
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(vars(args))
    cfg = opt

    set_seed(20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.get_device_name(0)}")

    identifier = f'{cfg["model"]}_{cfg["data_name"]}_{cfg["feature_pretrained"]}_flow{not cfg["no_flow"]}'
    result_path = create_outdir(osp.join(cfg["output_path"], identifier))
    
    # Ensure ckpts folder exists immediately
    os.makedirs(osp.join(result_path, "ckpts"), exist_ok=True)
    
    logger = get_logger(result_path)
    logger.info(cfg)

    print(">>> Building TEST loader")
    testloader = build_data_loader(cfg, mode="test")
    print(">>> TEST loader built")

    print(">>> Building TRAIN loader")
    trainloader = build_data_loader(cfg, mode="train")
    print(">>> TRAIN loader built")
    
    model = build_model(cfg, device)
    evaluate = build_eval(cfg)

    # --- EVALUATION MODE ---
    if args.eval is not None:
        model.load_state_dict(torch.load(args.eval))
        mAP = evaluate(model, testloader, logger, device)
        logger.info(f'{cfg["task"]} result: {mAP*100:.2f} m{cfg["metric"]}')
        exit()

    # --- TRAINING SETUP ---
    criterion = build_criterion(cfg, device)
    train_one_epoch = build_trainer(cfg)
    optim = torch.optim.AdamW if cfg["optimizer"] == "AdamW" else torch.optim.Adam
    optimizer = optim(
        [{"params": model.parameters(), "initial_lr": cfg["lr"]}],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = (
        build_lr_scheduler(cfg, optimizer, len(trainloader))
        if args.lr_scheduler
        else None
    )
    
    writer = SummaryWriter(osp.join(result_path, "runs")) if args.tensorboard else None
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f'Dataset: {cfg["data_name"]},  Model: {cfg["model"]}')
    logger.info(f'Total epoch:{cfg["num_epoch"]} | Total Params:{total_params/1e6:.1f} M')
    logger.info(f"Output Path:{result_path}")

    # --- AUTO-RESUME LOGIC ---
    start_epoch = 1
    best_mAP = 0
    best_epoch = 0
    
    # Check for 'checkpoint_last.pth' in the output folder
    resume_path = osp.join(result_path, "ckpts", "checkpoint_last.pth")
    if osp.exists(resume_path):
        print(f">>> Found checkpoint at {resume_path}. Resuming...")
        checkpoint = torch.load(resume_path)
        
        # Load Model
        model.load_state_dict(checkpoint['model'])
        # Load Optimizer (Crucial for AdamW momentum)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Load Scaler if exists
        if scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        
        # Restore variables
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint['best_mAP']
        best_epoch = checkpoint.get('best_epoch', 0)
        
        print(f">>> Resuming from Epoch {start_epoch} (Best mAP so far: {best_mAP*100:.2f})")
    else:
        print(">>> No checkpoint found. Starting from scratch.")


    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, cfg["num_epoch"] + 1):
        epoch_loss = train_one_epoch(
            trainloader,
            model,
            criterion,
            optimizer,
            scaler,
            epoch,
            device, 
            writer,
            scheduler=scheduler,
        )
        
        # Re-init features (specific to your code's logic)
        trainloader.dataset._init_features()
        
        # Evaluate
        mAP = evaluate(model, testloader, logger, device)
        print(f"Epoch {epoch} mAP: {mAP}")
        
        # Save Best
        if mAP > best_mAP:
            best_mAP = mAP
            best_epoch = epoch
            print(">>> New Best Model! Saving...")
            torch.save(model.state_dict(), osp.join(result_path, "ckpts", "best.pth"))
            
            logger.info(
                f'Epoch {epoch} mAP: {mAP*100:.2f} | Best mAP: {best_mAP*100:.2f} at epoch {best_epoch} | train_loss: {epoch_loss/len(trainloader):.4f}'
            )

        # --- SAFETY SAVE (Run every epoch) ---
        # 1. Save 'last' checkpoint (contains everything needed to resume)
        checkpoint_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_mAP': best_mAP,
            'best_epoch': best_epoch,
            'scaler': scaler.state_dict() if scaler else None
        }
        torch.save(checkpoint_state, osp.join(result_path, "ckpts", "checkpoint_last.pth"))
        
        # 2. Save specific epoch checkpoint (for your archive)
        torch.save(model.state_dict(), osp.join(result_path, "ckpts", f"checkpoint_epoch_{epoch}.pth"))
        print(f">>> Saved checkpoint_last.pth and checkpoint_epoch_{epoch}.pth")

    # Final Rename
    if osp.exists(osp.join(result_path, "ckpts", "best.pth")):
        os.rename(
            osp.join(result_path, "ckpts", "best.pth"),
            osp.join(result_path, "ckpts", f"best_{best_mAP*100:.2f}.pth"),
        )