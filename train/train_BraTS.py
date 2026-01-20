from data.dataset_load import *
from utils import *
# from network.network_add_nn_single import *
import importlib
import random, argparse, time, itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from accelerate import Accelerator
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import torchvision.utils as tv_utils
import logging
import torch.distributed as dist
from torchmetrics import MeanAbsoluteError 

NETWORK_CONFIGS = {
    'final': 'network.network',
    # Can continue to add more networks
}

def load_network_from_config(network_key):
    """Load network module from configuration"""
    if network_key not in NETWORK_CONFIGS:
        available_keys = list(NETWORK_CONFIGS.keys())
        raise ValueError(f"Unknown network '{network_key}'. Available: {available_keys}")
    
    module_path = NETWORK_CONFIGS[network_key]
    print(f"Loading network from module: {module_path}")
    try:
        module = importlib.import_module(module_path)
        TOMS = getattr(module, 'TOMS')
        Discriminator = getattr(module, 'Discriminator')
        return TOMS, Discriminator
    except ImportError as e:
        raise ImportError(f"Failed to import {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Required classes not found in {module_path}: {e}")

def set_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
def setup_logger(log_dir=None, log_name=None):
    """Simplified logging configuration"""
    # Automatically create log directory
    if log_dir is None:
        log_dir = os.path.join("logs",log_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("BraTS")
    logger.setLevel(logging.INFO)
    
    # Output to both file and console
    handlers = [
        logging.FileHandler(os.path.join(log_dir, "train.log")),
        logging.StreamHandler()
    ]
    
    # Unified log format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Add validation logging function
def log_validation_results(logger, epoch, recon_loss, psnr, ssim, msssim, mae):
    """Record validation metrics"""
    msg = (
        f"Epoch {epoch} | "
        f"Valid Recon Loss: {np.mean(recon_loss):.6f} | "
        f"PSNR: {np.mean(psnr):.6f} | "
        f"SSIM: {np.mean(ssim):.6f} | "
        f"MS-SSIM: {np.mean(msssim):.6f} | "
        f"MAE: {np.mean(mae):.6f} |"
    )
    logger.info(msg)
    
def main():
    parser = argparse.ArgumentParser(description='BraTS')

    parser.add_argument('--dataset', type=str, required=True,
                        help='path of training dataset')
    parser.add_argument('--identifier', type=str, required=True, metavar='N',
                        help='Select the identifier for file name')
    parser.add_argument('--batch-size', type=int,  default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--ch-dim', type=int,  default=64, metavar='N',
                        help='channel dimension for netwrok (default: 64)')
    parser.add_argument('--gradient_accumulation_steps', type=int,  default=1, metavar='N',
                        help='gradient_accumulation_steps for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epoches to train (default: 100)')
    parser.add_argument('--numlayers', type=int, default=4, metavar='N',
                        help='number of transformer layers(default: 4)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',
                        help='path of training snapshot')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--gpu', type=str, default='0', metavar='N',
                        help='Select the GPU (defualt 0)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='number of epoches to log (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--resume', action='store_true',
                        help='resume training by loading last snapshot')
    parser.add_argument('--debug_mode', action='store_true', help='Only load a small subset of data for debugging')
    parser.add_argument('--best_model',type=str, default='best_model',
                        help='path of training snapshot(best model)')
    parser.add_argument('--final_model',type=str, default='final_model',
                        help='path of training snapshot(final model)')
    parser.add_argument('--log_name', type=str, default='init',
                        help='name of the log file')
    parser.add_argument('--network', type=str, default='single',
                        choices=list(NETWORK_CONFIGS.keys()),
                        help=f'Network to use. Choices: {list(NETWORK_CONFIGS.keys())}')
    # parser.add_argument("--local_rank", type=int, default=0)  # Must add this parameter
    args = parser.parse_args()
    
    # dist.init_process_group(backend="nccl", init_method="env://")
    set_seed(args.seed)
    
    logger = setup_logger(log_name = args.log_name)
    best_model_path = os.path.join(args.checkpoints, args.identifier, args.best_model)
    final_model_path = os.path.join(args.checkpoints, args.identifier, args.final_model)
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path, exist_ok=True)
    if not os.path.exists(final_model_path):
        os.makedirs(final_model_path, exist_ok=True)
        
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    
    train_dataset = BraTSDataset(args.dataset, mode='train', debug_mode=args.debug_mode)
    valid_dataset = BraTSDataset(args.dataset, mode='valid', debug_mode=args.debug_mode)    
    
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, generator=generator)
    print(f"Load network: {args.network}")
    try:
        TOMS, Discriminator = load_network_from_config(args.network)
    except (ValueError, ImportError, AttributeError) as e:
        logger.error(f"Failed to load network: {e}")
        exit(1)
        
    model = TOMS(dim=args.ch_dim, num_inputs=4, num_outputs=1, dim_mults=(1,2,4,8,10), n_layers=args.numlayers, skip=True, blocks=False,batch_size=batch_size)
    discriminator = Discriminator(channels=1, num_filters_last=args.ch_dim)
    print("Model and Discriminator loaded successfully.")
    optimizer = Adam(model.parameters(), lr=0.0)
    optimizer_D = Adam(discriminator.parameters(), lr=0.0)
    steps_per_epoch = len(train_loader)
    total_iteration = epochs*steps_per_epoch*4
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=total_iteration, T_mult=1, eta_max=args.lr, T_up=100, gamma=0.5)
    scheduler_D = CosineAnnealingWarmUpRestarts(optimizer_D, T_0=total_iteration, T_mult=1, eta_max=args.lr*0.1, T_up=100, gamma=0.5)

    accelerator = Accelerator(gradient_accumulation_steps = args.gradient_accumulation_steps,
                              )
    
    device = accelerator.device
    
    valid_epochs = args.log_interval
    
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
    discriminator, optimizer_D, scheduler_D = accelerator.prepare(
        discriminator, optimizer_D, scheduler_D)
    print(f'Total Iteration: {total_iteration}')
    
    epoch = 0
    iterations = 0
    if args.resume:
        accelerator.load_state(input_dir=final_model_path)
        iterations = scheduler.scheduler.T_cur
        epoch = scheduler.scheduler.T_cur // steps_per_epoch 

    print(f'iteration: {iterations} epoch : {epoch}')
    
    loss_adversarial = torch.nn.BCEWithLogitsLoss()
    loss_auxiliary = torch.nn.CrossEntropyLoss()
    
    metric_psnr = PeakSignalNoiseRatio(data_range = 2.0).to(device)
    metric_siim = StructuralSimilarityIndexMeasure(data_range = 2.0).to(device)
    metric_mssiim = MultiScaleStructuralSimilarityIndexMeasure(data_range = 2.0).to(device)
    metric_mae = MeanAbsoluteError().to(device)
    
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.checkpoints, args.identifier, "logs", current_time)
    writer = SummaryWriter(log_dir)
    
    cand = [0, 1, 2, 3]
    candidates_all = []
    for L in range(len(cand) + 1):
        if L == 0 or L == 1:
            continue
        for subset in itertools.combinations(cand, L):
            candidates_all.append(subset)
    candidates = [list(filter(lambda x:m not in x, candidates_all)) for m in cand]
    max_SSIM = 0
    min_valid_con_loss = 1000000
    # accelerator = Accelerator()

    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    print(f"Number of processes: {accelerator.num_processes}")  
    print(f"Process index: {accelerator.process_index}")  
    print(f"Device: {accelerator.device}")
    #
    
    while epoch < epochs:
        epoch +=1
        avg_train_total_loss = []
        avg_train_adversarial_loss = []
        avg_train_auxiliary_loss = []
        avg_train_d_real_loss = []
        avg_train_d_fake_loss = []

   

        for n, batch in tqdm(enumerate(train_loader),colour="blue", total=len(train_loader)):
            # exit()
            # if accelerator.is_main_process:
            #         print(f"First batch sample on GPU 0: {batch['image'][0].mean()}")
            # if accelerator.process_index == 1:
            #         print(f"First batch sample on GPU 1: {batch['image'][0].mean()}")
            # input()
            with accelerator.accumulate(model):
                model.train()
                discriminator.train()
                
                inputs_all = batch['image'] # BxCxWxH
                targets = batch['target']
                modalities = batch['modalitiy'].squeeze(dim=-1) #（b,c,w*h）
                
                for m_shift in [False, True, True, True]:
                    iterations += 1
                    if m_shift:
                        for idx in range(inputs_all.shape[0]):
                            modalities[idx] = modalities[idx]+1 if modalities[idx]!=3 else 0
                            targets[idx,:] = inputs_all[idx,modalities[idx]:modalities[idx]+1,:]

                    targets_second = torch.zeros_like(targets, device=targets.device)
                    inputs_masked = -1*torch.ones_like(inputs_all, device=targets.device)
                    inputs_masked_second = -1*torch.ones_like(inputs_all, device=targets.device)
                    modalities_second = torch.zeros_like(modalities, device=modalities.device)

                    input_modals = [] # input_modals = [(0, 2, 3), (0, 1), (1, 2, 3), (2, 3), [1], [3], [0], [3]]
                    input_modals2 = [] # input_modals2 = [[0, 1, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2], [tensor(2)], [tensor(0)], [tensor(3)], [tensor(0)]]
                    input_modals_tensor =[]
                    input_modals2_tensor = []
                    for n, m in enumerate(modalities):# modalities = tensor([1, 2, 0, 0, 2, 0, 3, 0])
                        if n < inputs_all.shape[0] // 2: 
                            cand = random.choice(candidates[m])
                            cand_tensor = torch.tensor(cand, device=device)
                            input_modals.append(cand)
                            input_modals_tensor.append(cand_tensor)
                            for c in cand:
                                inputs_masked[n,c,:] = inputs_all[n,c,:]
                            m_masked = random.choice(cand)
                            modalities_second[n] = m_masked

                            cand2 = [x for x in [0,1,2,3] if x != m_masked]
                            cand2_tensor = torch.tensor(cand2, device=device)
                            input_modals2.append(cand2)
                            input_modals2_tensor.append(cand2_tensor)
                            
                            for c in cand2:
                                inputs_masked_second[n,c,:] = inputs_all[n,c,:]
                            targets_second[n,:] = inputs_all[n,m_masked:m_masked+1,:]
                        else:
                            cand = random.choice([x for x in [0,1,2,3] if x != m])
                            cand_tensor = torch.tensor([cand], device=device)
                            input_modals.append([cand])
                            
                            input_modals_tensor.append(cand_tensor)
                            inputs_masked[n,cand,:] = inputs_all[n,cand,:]
                            modalities_second[n] = cand
                            
                            cand2 = m
                            cand2_tensor = torch.tensor([cand2], device=device)
                            input_modals2.append([cand2])
                            input_modals2_tensor.append(cand2_tensor)
                            inputs_masked_second[n,cand2,:] = inputs_all[n,cand2,:]
                            targets_second[n,:] = inputs_all[n,cand:cand+1,:]

                    # train G
                    optimizer.zero_grad()
                    model_encoder = model.module.encoder if hasattr(model, "module") else model.encoder
                    model_middle = model.module.middle if hasattr(model, "module") else model.middle
                    model_decoder = model.module.decoder if hasattr(model, "module") else model.decoder
                    
                    
                    f, h = model_encoder(inputs_masked, input_modals, input_modals_tensor, modalities, train_mode=True)
                    z = model_middle(f, modalities)
                    targets_recon = model_decoder(z, h)

                    recon_loss = (torch.abs(targets - targets_recon)).mean()

                    for n, m in enumerate(modalities):
                        inputs_masked_second[n,m,:] = targets_recon[n,0,:]
                    f_recon, h_recon = model_encoder(inputs_masked_second, input_modals2,input_modals2_tensor,modalities_second, train_mode=True)
                    feature_l1_loss = 1 - F.cosine_similarity(f.flatten(1,-1), f_recon.flatten(1,-1)).mean()
                    
                    z_recon = model_middle(f_recon, modalities_second)

                    targets_cycle = model_decoder(z_recon, h_recon)
                    cycle_loss = (torch.abs(targets_second - targets_cycle)).mean()

                    logits_fake, labels_fake = discriminator(targets_recon)
                    # If GPU is available, use CUDA
                    if torch.cuda.is_available():
                        valid=torch.ones(logits_fake.shape).cuda()
                        fake=torch.zeros(logits_fake.shape).cuda()
                    else:
                        valid=torch.ones(logits_fake.shape).cpu()
                        fake=torch.zeros(logits_fake.shape).cpu()
                   

                    adversarial_loss = loss_adversarial(logits_fake, valid)
                    auxiliary_loss = loss_auxiliary(labels_fake, modalities)

                    total_loss = 10*recon_loss +0.25*adversarial_loss + 0.25*auxiliary_loss + 1*feature_l1_loss + 1*cycle_loss

                    accelerator.backward(total_loss)
                    optimizer.step()

                    optimizer_D.zero_grad()

                    logits_real, labels_real = discriminator(targets_second)
                    logits_fake, labels_fake = discriminator(targets_recon.detach())
                    
                    d_real_adv = loss_adversarial(logits_real, valid)
                    d_fake_adv = loss_adversarial(logits_fake, fake)
                    
                    d_real_aux = loss_auxiliary(labels_real, modalities_second)
                    d_fake_aux = loss_auxiliary(labels_fake, modalities)

                    d_loss = 0.25*(d_real_adv + d_fake_adv) + 0.25*d_real_aux + 0.25*d_fake_aux
                    accelerator.backward(d_loss)
                    
                    optimizer_D.step()

                    scheduler.step(iterations)
                    scheduler_D.step(iterations)
                    
                    avg_train_total_loss.append(total_loss.item())
                    avg_train_adversarial_loss.append(adversarial_loss.item())
                    avg_train_auxiliary_loss.append(auxiliary_loss.item())
                    avg_train_d_real_loss.append(d_real_adv.item())
                    avg_train_d_fake_loss.append(d_fake_adv.item())
                    
        avg_train_total_loss_T = torch.tensor(avg_train_total_loss, device=device)
        avg_train_adversarial_loss_T = torch.tensor(avg_train_adversarial_loss, device=device)
        avg_train_auxiliary_loss_T = torch.tensor(avg_train_auxiliary_loss, device=device)
        avg_train_d_real_loss_T = torch.tensor(avg_train_d_real_loss, device=device)
        avg_train_d_fake_loss_T = torch.tensor(avg_train_d_fake_loss, device=device)

        avg_train_total_loss_T_gather = accelerator.gather(avg_train_total_loss_T)
        avg_train_adversarial_loss_T_gather = accelerator.gather(avg_train_adversarial_loss_T)
        avg_train_auxiliary_loss_T_gather = accelerator.gather(avg_train_auxiliary_loss_T)
        avg_train_d_real_loss_T_gather = accelerator.gather(avg_train_d_real_loss_T)
        avg_train_d_fake_loss_T_gather = accelerator.gather(avg_train_d_fake_loss_T)
                    
              
        train_total_loss_result = avg_train_total_loss_T_gather.mean().item()
        train_adversarial_loss_result = avg_train_adversarial_loss_T_gather.mean().item()
        train_auxiliary_loss_result = avg_train_auxiliary_loss_T_gather.mean().item()
        train_d_real_loss_result = avg_train_d_real_loss_T_gather.mean().item()
        train_d_fake_loss_result = avg_train_d_fake_loss_T_gather.mean().item()

        
        if accelerator.is_main_process:
            global_step = epoch * len(train_loader) + n
            writer.add_scalar('Loss/Total', train_total_loss_result, global_step)
            writer.add_scalar('Loss/Adversarial', train_adversarial_loss_result, global_step)
            writer.add_scalar('Loss/Auxiliary', train_auxiliary_loss_result, global_step)
            writer.add_scalar('Loss/D_Real', train_d_real_loss_result, global_step)
            writer.add_scalar('Loss/D_Fake', train_d_fake_loss_result, global_step)
    
            logger.info(
                f"Epoch {epoch} | "
                f"Train Loss: {train_total_loss_result:.6f} | "
                f"G Loss: {train_adversarial_loss_result+train_auxiliary_loss_result:.6f} | "
                f"D Loss: {train_d_real_loss_result+train_d_fake_loss_result:.6f} | "

            )

        if epoch % valid_epochs == 0:
            accelerator.save_state(output_dir=final_model_path)  
            with torch.no_grad():
                avg_valid_recon_loss = []
                avg_valid_psnr = []
                avg_valid_ssim = []
                avg_valid_msssim = []
                avg_valid_mae = []



                for batch in tqdm(valid_loader,colour="green"):
                    model.eval()
                    
                    inputs = batch['image_masked'] # BxCxWxH
                    targets = batch['target']
                    modalities = batch['modalitiy'].squeeze(dim=-1)
                    recon_list = []
                    input_modals = []
                    inputs_modals_tensor = []
                   
               
                    for n, m in enumerate(modalities):
                        cand = [x for x in [0,1,2,3] if x != m]
                        input_modals.append(cand)
                        inputs_modals_tensor.append(torch.tensor(cand, device=device))
                    model_encoder = model.module.encoder if hasattr(model, "module") else model.encoder
                    model_middle = model.module.middle if hasattr(model, "module") else model.middle
                    model_decoder = model.module.decoder if hasattr(model, "module") else model.decoder
                    
                    f, h = model_encoder(inputs, input_modals,inputs_modals_tensor, modalities)
                    z = model_middle(f, modalities)
                    targets_recon = model_decoder(z, h)
                    recon_list.append(targets_recon.cpu().numpy())

                    for j in range(targets_recon.shape[0]):
                        avg_valid_recon_loss.append(torch.abs(targets[j:j+1,:] - targets_recon[j:j+1,:]).mean().cpu())
                        avg_valid_psnr.append(metric_psnr(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu())
                        avg_valid_ssim.append(metric_siim(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu())
                        avg_valid_msssim.append(metric_mssiim(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu())
                        avg_valid_mae.append(metric_mae(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu())
                    
                avg_valid_recon_loss_T = torch.tensor(avg_valid_recon_loss, device=device)
                avg_valid_psnr_T = torch.tensor(avg_valid_psnr, device=device)
                avg_valid_ssim_T = torch.tensor(avg_valid_ssim, device=device)
                avg_valid_msssim_T = torch.tensor(avg_valid_msssim, device=device)
                avg_valid_mae_T = torch.tensor(avg_valid_mae, device=device)

                avg_valid_recon_loss_T_gather = accelerator.gather(avg_valid_recon_loss_T)
                avg_valid_psnr_T_gather = accelerator.gather(avg_valid_psnr_T)
                avg_valid_ssim_T_gather = accelerator.gather(avg_valid_ssim_T)
                avg_valid_msssim_T_gather = accelerator.gather(avg_valid_msssim_T)
                avg_valid_mae_T_gather = accelerator.gather(avg_valid_mae_T)

                all_avg_valid_recon_loss = avg_valid_recon_loss_T_gather.mean().item()
                all_avg_valid_psnr = avg_valid_psnr_T_gather.mean().item()
                all_avg_valid_ssim = avg_valid_ssim_T_gather.mean().item()
                all_avg_valid_msssim = avg_valid_msssim_T_gather.mean().item()
                all_avg_valid_mae = avg_valid_mae_T_gather.mean().item()

                if accelerator.is_main_process:
                    writer.add_scalar('Epoch_Metric/Valid_Recon', all_avg_valid_recon_loss, epoch)
                    writer.add_scalar('Epoch_Metric/Valid_PSNR', all_avg_valid_psnr, epoch)
                    writer.add_scalar('Epoch_Metric/Valid_SSIM', all_avg_valid_ssim, epoch)
                    writer.add_scalar('Epoch_Metric/Valid_MS_SSIM', all_avg_valid_msssim, epoch)
                    writer.add_scalar('Epoch_Metric/Valid_MAE', all_avg_valid_mae, epoch)
    
                    log_validation_results(logger, epoch, all_avg_valid_recon_loss, all_avg_valid_psnr, all_avg_valid_ssim, all_avg_valid_msssim, all_avg_valid_mae)
    
                    cur_valid_con_loss = all_avg_valid_recon_loss
                    if cur_valid_con_loss < min_valid_con_loss:
                        min_valid_con_loss = cur_valid_con_loss
                        cur_best_model_path = os.path.join(best_model_path, f"epoch_{epoch}")
                        if not os.path.exists(cur_best_model_path):
                            os.makedirs(cur_best_model_path, exist_ok=True)
                        accelerator.save_state(output_dir=best_model_path) 
                        logger.info(f"save model to {best_model_path}")

                    
        writer.close()
        logging.shutdown()       

if __name__ == "__main__":
    main()
    