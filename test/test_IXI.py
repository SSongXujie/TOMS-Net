from data.dataset_load import *
from utils import *
# from network.network_real_fusion import *

import random, argparse, time, itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from accelerate import Accelerator
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm
import datetime
import os
from PIL import Image
import torch
import importlib
from config.get_argparser import get_argparser, NETWORK_CONFIGS
NETWORK_CONFIGS = {
    'final': 'network.network',
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

def normalize_batch_vectorized(tensor, target_min=-1, target_max=1):
    """Normalize batch data using vectorized operations"""
    batch_size = tensor.shape[0]

    tensor_flat = tensor.view(batch_size, -1)

    sample_min = tensor_flat.min(dim=1, keepdim=True)[0]  # (batch_size, 1)
    sample_max = tensor_flat.max(dim=1, keepdim=True)[0]  # (batch_size, 1)

    original_shape = tensor.shape
    sample_min = sample_min.view(batch_size, *[1] * (len(original_shape) - 1))
    sample_max = sample_max.view(batch_size, *[1] * (len(original_shape) - 1))

    range_mask = (sample_max != sample_min)
    normalized = torch.where(
        range_mask,
        (tensor - sample_min) / (sample_max - sample_min + 1e-8) * (target_max - target_min) + target_min,
        torch.full_like(tensor, (target_min + target_max) / 2)
    )
    
    return normalized

def generate_masked_batch(inputs, device, masked_modals, available_modals):
    batch_size, num_inputs, height, width = inputs.shape

    actual_batch_size = inputs.shape[0]
    masked_modals = torch.tensor([masked_modals], dtype=torch.long)  
    masked_modals_batch = masked_modals.repeat(actual_batch_size) 
    input_modals = []
    inputs_modals_tensor = []
    inputs_masked = -1*torch.ones_like(inputs) 
    for i in range(batch_size):
        available_modals = available_modals.copy() 
        # [x for x in range(4) if x != m]
        selected_modals = available_modals
        input_modals.append(selected_modals)
        selected_modals_tensor = torch.tensor(selected_modals, dtype=torch.long,device=device)  
        inputs_modals_tensor.append(selected_modals_tensor)
        for modal in selected_modals:
            inputs_masked[i, modal, ...] = inputs[i, modal, ...]
    
    targets = inputs[:, masked_modals, ...] 
    
    return inputs_masked.to(device), targets.to(device), masked_modals_batch.to(device), input_modals, inputs_modals_tensor

def set_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
def process_and_save_sample(target, recon,inputs, save_dir, batch_idx):
    """Process and save a single sample"""
    target_np = target.squeeze().numpy() 
    recon_np = recon.squeeze().numpy()
    inputs_np = inputs.squeeze().numpy()  
    available_modals_num = inputs_np.shape[0]

    

    def normalize(img):
        img = img.astype(np.float32)
        current_min = np.min(img)
        current_max = np.max(img)
        return ((img - current_min) / (current_max - current_min + 1e-8) * 255).astype(np.uint8)
    
    target_norm = normalize(target_np)
    recon_norm = normalize(recon_np)
    tensor_inputs = []
    for i in range(available_modals_num):
        tensor_i = torch.tensor([i], dtype=torch.long) 
        tensor_in = normalize(inputs_np[tensor_i,...])
        tensor_inputs.append(tensor_in)       

    comparison = Image.new('RGB', (target_norm.shape[1]*2, target_norm.shape[0]))
    comparison.paste(Image.fromarray(target_norm, mode='L'), (0, 0))
    comparison.paste(Image.fromarray(recon_norm, mode='L'), (target_norm.shape[1], 0))

    base_name = f"sample_{batch_idx:04d}"
    comparison.save(os.path.join(save_dir, f"{base_name}_comparison.png"))
    
    rgb_img = Image.new('RGB', (inputs_np[0].shape[1]*available_modals_num, inputs_np[0].shape[0]))
    for i in range(available_modals_num):
        input_img = Image.fromarray(tensor_inputs[i], mode='L')
        rgb_img.paste(input_img, (inputs_np[0].shape[1] * i, 0))
    rgb_img.save(os.path.join(save_dir, f"{base_name}_input.png"))
    # input("Press Enter to continue...")
   
def save_all_inputs(inputs, save_dir, batch_idx):
    """Process and save a single sample"""
    inputs_np = inputs.squeeze().numpy()
    available_modals_num = inputs_np.shape[0]
    def normalize(img):
        img = img.astype(np.float32)
        current_min = np.min(img)
        current_max = np.max(img)
        return ((img - current_min) / (current_max - current_min + 1e-8) * 255).astype(np.uint8)
    
    for i in range(available_modals_num):
        inputs_np[i] = normalize(inputs_np[i])

    rgb_img = Image.new('RGB', (inputs_np[0].shape[1]*available_modals_num, inputs_np[0].shape[0]))
    for i in range(available_modals_num):
        input_img = Image.fromarray(inputs_np[i], mode='L')
        rgb_img.paste(input_img, (inputs_np[0].shape[1] * i, 0))

    base_name = f"sample_{batch_idx:04d}"
    rgb_img.save(os.path.join(save_dir, f"{base_name}_input.png")) 
    
def main():
    parser = argparse.ArgumentParser(description='BraTS Testing')

    parser.add_argument('--dataset', type=str, required=True,
                        help='path of testing dataset')
    # parser.add_argument('--identifier', type=str, required=True, metavar='N',
    #                     help='Select the identifier for file name to load model')
    parser.add_argument('--batch-size', type=int,  default=6, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--ch-dim', type=int,  default=64, metavar='N',
                        help='channel dimension for network (default: 64)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--gpu', type=str, default='5', metavar='N',
                        help='Select the GPU (default 0)')
    parser.add_argument('--numlayers', type=int, default=4, metavar='N',
                        help='number of transformer layers(default: 4)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_1',
                        help='path of trained model checkpoints')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                        help='path to save test results')
    parser.add_argument('--masked_modals', type=int, default=0,
                        help='masked modals for testing')
    parser.add_argument('--save_results_name', type=str, default='hhh',)
    parser.add_argument('--debug_mode', action='store_true', help='Only load a small subset of data for debugging')
    parser.add_argument('--save_images', action='store_true', help='Save images for visualization')
    parser.add_argument('--network', type=str, default='single',
                        choices=list(NETWORK_CONFIGS.keys()),
                        help=f'Network to use. Choices: {list(NETWORK_CONFIGS.keys())}')

    args = parser.parse_args()

    set_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    batch_size = args.batch_size

    accelerator = Accelerator()
    device = accelerator.device
    masked_modals = device.index
    print(f"Using device: {device}")
    cand = [x for x in range(3) if x != masked_modals]
    available_modals_list = []
    candidates = []
    for L in range(1, len(cand)+1):
        for subset in itertools.combinations(cand, L):
            available_modals_list.append(subset)

    print(cand)
    print(available_modals_list)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    test_dataset = BraTSDataset(args.dataset, mode='test',debug_mode=args.debug_mode) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    print(f"Load network: {args.network}")
    try:
        TOMS, _ = load_network_from_config(args.network)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"Error loading network: {e}")
        exit(1)
    
    model = TOMS(dim=args.ch_dim, num_inputs=3, num_outputs=1, dim_mults=(1,2,4,8,10), n_layers=args.numlayers, skip=True, blocks=False,image_size=256,H=256,W=256,batch_size=batch_size)

    
    
    model = accelerator.prepare(model)
    
    # Load trained model
    accelerator.load_state(input_dir=args.checkpoints)
    
    metric_psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    metric_msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    model_encoder = model.module.encoder if hasattr(model, "module") else model.encoder
    model_middle = model.module.middle if hasattr(model, "module") else model.middle
    model_decoder = model.module.decoder if hasattr(model, "module") else model.decoder
    
    for available_modals in available_modals_list:
        with torch.no_grad():
            test_recon_loss = []
            test_psnr = []
            test_ssim = []
            test_msssim = []
            recon_images = []
            target_images = []
            available_modals = list(available_modals)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(args.output_dir, args.save_results_name, f"{available_modals}->{masked_modals}",f"{timestamp}", "saved_images")
            print(f"save_dir: {save_dir}")
            # exit()
            os.makedirs(save_dir, exist_ok=True)
            
            results_text_file = os.path.join(args.output_dir, args.save_results_name, f"{available_modals}->{masked_modals}",f"{timestamp}", f"{available_modals}_to_{masked_modals}_result.txt")
            
            result_file = os.path.join(args.output_dir,args.save_results_name,f"{available_modals}->{masked_modals}",f"{timestamp}","saved_pt")
            os.makedirs(result_file, exist_ok=True)
            total_samples = 0
            processed_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing", colour="green")):
                model.eval()
                # if len(batch)<6:
                #     continue
                inputs = batch['image'].to(device)  # BxCxWxH
                image_masked, targets, modalities, input_modals,input_modals_tensor =generate_masked_batch(inputs, device, masked_modals, available_modals)
                f, h = model_encoder(image_masked, input_modals,input_modals_tensor, modalities)
                z = model_middle(f, modalities)
                targets_recon = model_decoder(z, h)
                if args.save_images:
                    process_and_save_sample(
                    targets[0].cpu(),
                    targets_recon[0].cpu(),
                    inputs[0].cpu(),
                    save_dir,
                    batch_idx
                    )
                recon_images.append(targets_recon.cpu())
                target_images.append(targets.cpu())
                
                # Calculate metrics
                for j in range(targets_recon.shape[0]):
                    test_recon_loss.append(torch.abs(targets[j:j+1,:] - targets_recon[j:j+1,:]).mean().cpu())
                    test_psnr.append(metric_psnr(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu())
                    test_ssim.append(metric_ssim(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu())
                    test_msssim.append(metric_msssim(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu())
            
            # Save results
            results = {
                'recon_loss': torch.mean(torch.stack(test_recon_loss)).item(),
                'psnr_mean': torch.mean(torch.stack(test_psnr)).item(),
                'psnr_std': torch.std(torch.stack(test_psnr),unbiased=False).item(),
                'ssim_mean': torch.mean(torch.stack(test_ssim)).item(),
                'ssim_std': torch.std(torch.stack(test_ssim),unbiased=False).item(),
                'msssim_mean': torch.mean(torch.stack(test_msssim)).item(),
                'msssim_std': torch.std(torch.stack(test_msssim),unbiased=False).item(),
                'recon_images': torch.cat(recon_images, dim=0),
                'target_images': torch.cat(target_images, dim=0)
            }
            if not os.path.exists(results_text_file):
                with open(results_text_file, 'w') as f:
                    f.write(f"Available Modals: {available_modals}\n")
                    f.write(f"Masked Modals: {masked_modals}\n")
                    f.write(f"Recon Loss: {results['recon_loss']:.6f}\n")
                    f.write(f"PSNR: {results['psnr_mean']:.6f}\n")
                    f.write(f"SSIM: {results['ssim_mean']:.6f}\n")
                    f.write(f"MS-SSIM: {results['msssim_mean']:.6f}\n")
                    f.write(f"PSNR Std: {results['psnr_std']:.6f}\n")
                    f.write(f"SSIM Std: {results['ssim_std']:.6f}\n")
                    f.write(f"MS-SSIM Std: {results['msssim_std']:.6f}\n")
                    # f.write(f"Total Samples: {len(test_recon_loss
                    
            # Save results to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            result_file = os.path.join(result_file, f"test_results_{timestamp}.pt")
            torch.save(results, result_file)
            
            print("\nTest Results:")
            print(f"{available_modals}->{masked_modals}")
            print(f"Recon Loss: {results['recon_loss']:.6f}")
            print(f"PSNR: {results['psnr_mean']:.6f}")
            print(f"SSIM: {results['ssim_mean']:.6f}")
            print(f"MS-SSIM: {results['msssim_mean']:.6f}")
            print(f"PSNR Std: {results['psnr_std']:.6f}")
            print(f"SSIM Std: {results['ssim_std']:.6f}")
            print(f"MS-SSIM Std: {results['msssim_std']:.6f}")
            print(f"\nResults saved to {result_file}")

if __name__ == "__main__":
    main()