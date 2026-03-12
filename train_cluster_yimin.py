import os
import pdb
import numpy as np
from pathlib import Path
import torch
import random
import argparse
import time
from torchvision import transforms
import rasterio

# Import your existing utilities
#import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss
from utils.evaluation import Evaluator
from utils.setup_logger import setup_logger
#from utils.visual_predict import visualize_predict
from models.mamba_cluster_hackathon14 import cluster_MambaHSI as MambaHSI

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.autograd.set_detect_anomaly(True)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/train_val_test_patches/patches')
    parser.add_argument('--clusters_root', type=str, default='/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/clustered_datasets_original')
    parser.add_argument('--sensor', type=str, default='sentinel2', choices=['sentinel2'])
    parser.add_argument('--label_type', type=str, default='filtered', choices=['filtered', 'unfiltered'])
    parser.add_argument('--work_dir', type=str, default='./')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='RUNS')
    parser.add_argument('--num_classes', type=int, default=13)
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    return args

# random seed setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_parser()

exp_name = args.exp_name
seed_list = [args.seed]

max_epoch = args.max_epoch
learning_rate = args.lr
net_name = 'MambaHSI'

paras_dict = {'net_name': net_name, 'lr': learning_rate, 'seed_list': seed_list}

# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

def pad_cluster_counts(cluster_map, target_clusters):
    unique, counts = np.unique(cluster_map, return_counts=True)
    padded = np.zeros(target_clusters, dtype=int)
    # Only fill clusters that exist (cluster IDs 0 to max_cluster)
    max_cluster = min(len(padded) - 1, unique.max())
    valid_indices = unique[unique <= max_cluster]
    valid_counts = counts[unique <= max_cluster]
    padded[valid_indices] = valid_counts
    return padded.tolist()

if __name__ == '__main__':
    data_root = Path(args.data_root)
    clusters_root = Path(args.clusters_root)
    work_dir = args.work_dir
    
    dataset_name = f"{args.sensor}_{args.label_type}"
    
    save_folder = os.path.join(work_dir, exp_name, net_name, dataset_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_log_path = os.path.join(save_folder, 'train.log')
    logger = setup_logger(name=dataset_name, logfile=save_log_path)
    torch.cuda.empty_cache()
    
    logger.info(save_folder)
    logger.info(paras_dict)
    
    class_count = args.num_classes
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    
    for exp_idx, curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)
        single_experiment_name = f'run{exp_idx}_seed{curr_seed}'
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis')
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
        
        save_weight_path = os.path.join(save_single_experiment_folder, "best_model.pth")
        results_save_path = os.path.join(save_single_experiment_folder, 'result.txt')
        predict_save_path = os.path.join(save_single_experiment_folder, 'pred_vis.png')
        gt_save_path = os.path.join(save_single_experiment_folder, 'gt_vis.png')
        
        # Build Model - EXACTLY as in original
        # We need to know the channel count - load first patch to check
        first_patch_path = data_root / 'train' / args.sensor / 'img' / 'class_0_patch_0.tif'
        with rasterio.open(first_patch_path) as src:
            first_patch = src.read()
            channels = first_patch.shape[0]
        
        # Use fixed cluster numbers as in original code
        # Original used: 50, 30, 20 clusters for the three scales
        num_clusters = 50  # This is for scale 100, will be divided internally
        
        net = MambaHSI(
            in_channels=channels, 
            num_classes=class_count, 
            hidden_dim=128, 
            num_clusters=num_clusters,  # Same as original
            sparsity_ratio=1.0
        )
        
        net.to(device)
        logger.info(net)
        
        # Prepare data loaders for each split
        def create_data_loader(split):
            img_dir = data_root / split / args.sensor / 'img'
            label_dir = data_root / split / 'labels' / args.label_type
            clusters_dir = clusters_root / args.sensor / args.label_type / split / 'clusters'
            
            # Get all patches
            patches = []
            for img_path in sorted(img_dir.glob('*.tif')):
                patch_name = img_path.stem
                label_path = label_dir / f"{patch_name}.tif"
                
                if label_path.exists():
                    patches.append((img_path, label_path, patch_name))
            
            return patches, clusters_dir
        
        # Get patches for each split
        train_patches, train_clusters_dir = create_data_loader('train')
        val_patches, val_clusters_dir = create_data_loader('val')
        test_patches, test_clusters_dir = create_data_loader('test')
        
        logger.info(f"Train patches: {len(train_patches)}")
        logger.info(f"Val patches: {len(val_patches)}")
        logger.info(f"Test patches: {len(test_patches)}")
        
        # Training loop over epochs
        train_loss_list = [100]
        train_acc_list = [0]
        val_loss_list = [100]
        val_acc_list = [0]
        
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        logger.info(optimizer)
        
        best_val_acc = 0
        
        for epoch in range(max_epoch):
            net.train()
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle training patches each epoch
            random.shuffle(train_patches)
            
            # Process each training patch
            for img_path, label_path, patch_name in train_patches:
                try:
                    # Load image
                    with rasterio.open(img_path) as src:
                        image = src.read()  # (C, H, W)
                    # LOAD LABEL - YOU'RE MISSING THIS!
                    with rasterio.open(label_path) as src:
                        label = src.read(1)  # (H, W)
                        label = np.where(label == -99, -1, label)  # Convert background
                   
                    # Load cluster maps
                    cluster100 = np.load(train_clusters_dir / f"{patch_name}_cluster100.npy")
                    cluster50 = np.load(train_clusters_dir / f"{patch_name}_cluster50.npy")
                    cluster30 = np.load(train_clusters_dir / f"{patch_name}_cluster30.npy")
                    
                    print(f"Cluster100: unique={np.unique(cluster100)}, max={cluster100.max()}")
                    print(f"Cluster50: unique={np.unique(cluster50)}, max={cluster50.max()}")
                    print(f"Cluster30: unique={np.unique(cluster30)}, max={cluster30.max()}")
                    # Convert to tensors
                    image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).to(device)
                    label_tensor = torch.from_numpy(label).unsqueeze(0).long().to(device)
                    cluster100_tensor = torch.from_numpy(cluster100).long().to(device)
                    cluster50_tensor = torch.from_numpy(cluster50).long().to(device)
                    cluster30_tensor = torch.from_numpy(cluster30).long().to(device)
                    
                    # Get per_cluster_num from cluster maps (same as original)
                    num_label100 = pad_cluster_counts(cluster100, 50)  # Always 50 clusters
                    num_label50 = pad_cluster_counts(cluster50, 30)    # Always 30 clusters  
                    num_label30 = pad_cluster_counts(cluster30, 20)    # Always 20 clusters
                    per_cluster_num = [num_label100, num_label50, num_label30]
                    
                    # Forward pass (EXACTLY as in original)
                    y_pred, loss2 = net(
                        image_tensor, 
                        per_cluster_num, 
                        cluster100_tensor, 
                        cluster50_tensor, 
                        cluster30_tensor
                    )
                    
                    # Compute loss (EXACTLY as in original)
                    ls = head_loss(loss_func, y_pred, label_tensor) + loss2
                    
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()
                    
                    epoch_loss += ls.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing {patch_name}: {e}")
            
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(f'Iter:{epoch}|loss:{avg_epoch_loss}')
            
            # Validation (every 50 epochs like original)
            if (epoch + 1) % 1 == 0:
                net.eval()
                evaluator = Evaluator(num_class=class_count)
                
                with torch.no_grad():
                    val_predictions = []
                    val_labels = []
                    
                    for img_path, label_path, patch_name in val_patches:
                        try:
                            # Load validation data
                            with rasterio.open(img_path) as src:
                                image = src.read()
                            with rasterio.open(label_path) as src:
                                label = src.read(1)
                                label = np.where(label == -99, -1, label)
                            
                            # Load cluster maps
                            cluster100 = np.load(val_clusters_dir / f"{patch_name}_cluster100.npy")
                            cluster50 = np.load(val_clusters_dir / f"{patch_name}_cluster50.npy")
                            cluster30 = np.load(val_clusters_dir / f"{patch_name}_cluster30.npy")
                            
                            # Convert to tensors
                            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).to(device)
                            label_tensor = torch.from_numpy(label).unsqueeze(0).long().to(device)
                            cluster100_tensor = torch.from_numpy(cluster100).long().to(device)
                            cluster50_tensor = torch.from_numpy(cluster50).long().to(device)
                            cluster30_tensor = torch.from_numpy(cluster30).long().to(device)
                            
                            # Get per_cluster_num
                            num_label100 = np.unique(cluster100, return_counts=True)[1].tolist()
                            num_label50 = np.unique(cluster50, return_counts=True)[1].tolist()
                            num_label30 = np.unique(cluster30, return_counts=True)[1].tolist()
                            per_cluster_num = [num_label100, num_label50, num_label30]
                            
                            # Forward pass
                            output_val = net(
                                image_tensor, 
                                per_cluster_num, 
                                cluster100_tensor, 
                                cluster50_tensor, 
                                cluster30_tensor
                            )
                            
                            seg_logits = output_val
                            predict = torch.argmax(seg_logits, dim=1).cpu().numpy()
                            Y_val_np = label_tensor.cpu().numpy()
                            Y_val_255 = np.where(Y_val_np == -1, 255, Y_val_np)
                            
                            evaluator.add_batch(Y_val_255, predict)
                            
                        except Exception as e:
                            logger.warning(f"Error in validation {patch_name}: {e}")
                
                # Compute metrics
                OA = evaluator.Pixel_Accuracy()
                mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                Kappa = evaluator.Kappa()
                
                logger.info(f'Evaluate {epoch}|OA:{OA}|MACC:{mAcc}|Kappa:{Kappa}|MIOU:{mIOU}')
                
                # Save best model
                if OA >= best_val_acc:
                    best_epoch = epoch + 1
                    best_val_acc = OA
                    torch.save(net.state_dict(), save_weight_path)
                    logger.info(f"saved weights at epoch {epoch}")
        
        # Testing with best model
        logger.info("\n\n====================Starting evaluation for testing set.========================")
        
        best_net = MambaHSI(
            in_channels=channels, 
            num_classes=class_count, 
            hidden_dim=128, 
            num_clusters=num_clusters,
            sparsity_ratio=1.0
        )
        
        best_net.to(device)
        best_net.load_state_dict(torch.load(save_weight_path))
        best_net.eval()
        
        test_evaluator = Evaluator(num_class=class_count)
        
        with torch.no_grad():
            test_predictions = []
            test_labels = []
            
            for img_path, label_path, patch_name in test_patches:
                try:
                    # Load test data
                    with rasterio.open(img_path) as src:
                        image = src.read()
                    with rasterio.open(label_path) as src:
                        label = src.read(1)
                        label = np.where(label == -99, -1, label)
                    
                    # Load cluster maps
                    cluster100 = np.load(test_clusters_dir / f"{patch_name}_cluster100.npy")
                    cluster50 = np.load(test_clusters_dir / f"{patch_name}_cluster50.npy")
                    cluster30 = np.load(test_clusters_dir / f"{patch_name}_cluster30.npy")
                    
                    # Convert to tensors
                    image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).to(device)
                    label_tensor = torch.from_numpy(label).unsqueeze(0).long().to(device)
                    cluster100_tensor = torch.from_numpy(cluster100).long().to(device)
                    cluster50_tensor = torch.from_numpy(cluster50).long().to(device)
                    cluster30_tensor = torch.from_numpy(cluster30).long().to(device)
                    
                    # Get per_cluster_num
                    num_label100 = np.unique(cluster100, return_counts=True)[1].tolist()
                    num_label50 = np.unique(cluster50, return_counts=True)[1].tolist()
                    num_label30 = np.unique(cluster30, return_counts=True)[1].tolist()
                    per_cluster_num = [num_label100, num_label50, num_label30]
                    
                    # Forward pass
                    output_test = best_net(
                        image_tensor, 
                        per_cluster_num, 
                        cluster100_tensor, 
                        cluster50_tensor, 
                        cluster30_tensor
                    )
                    
                    seg_logits_test = output_test
                    predict_test = torch.argmax(seg_logits_test, dim=1).cpu().numpy()
                    Y_test_np = label_tensor.cpu().numpy()
                    Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)
                    
                    test_evaluator.add_batch(Y_test_255, predict_test)
                    
                except Exception as e:
                    logger.warning(f"Error in testing {patch_name}: {e}")
        
        # Compute test metrics
        OA_test = test_evaluator.Pixel_Accuracy()
        mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
        mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
        Kappa_test = test_evaluator.Kappa()
        
        logger.info(f'Test |OA:{OA_test}|MACC:{mAcc_test}|Kappa:{Kappa_test}|MIOU:{mIOU_test}')
        
        # Save results
        f = open(results_save_path, 'a+')
        str_results = f'\n======================' \
                      f' seed={curr_seed}' \
                      f' learning rate={learning_rate}' \
                      f' epochs={max_epoch}' \
                      f' ======================' \
                      f'\nOA={OA_test}' \
                      f'\nAA={mAcc_test}' \
                      f'\nkpp={Kappa_test}' \
                      f'\nmIOU_test:{mIOU_test}' \
                      f'\nIOU_test:{IOU_test}' \
                      f'\nAcc_test:{Acc_test}\n'
        logger.info(str_results)
        f.write(str_results)
        f.close()
        
        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)
        
        torch.cuda.empty_cache()
    
    # Save mean results
    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    
    np.set_printoptions(precision=4)
    logger.info(f"\n====================Mean result of {len(seed_list)} times runs =========================")
    logger.info(f'OA={round(np.mean(OA_ALL) * 100, 2)} +- {round(np.std(OA_ALL) * 100, 2)}')
    logger.info(f'AA={round(np.mean(AA_ALL) * 100, 2)} +- {round(np.std(AA_ALL) * 100, 2)}')
    logger.info(f'Kpp={round(np.mean(KPP_ALL) * 100, 2)} +- {round(np.std(KPP_ALL) * 100, 2)}')
    
    mean_result_path = os.path.join(save_folder, 'mean_result.txt')
    with open(mean_result_path, 'w') as f:
        f.write(f'\n\n***************Mean result of {len(seed_list)} times runs ********************' \
                f'\nOA={round(np.mean(OA_ALL) * 100, 2)} +- {round(np.std(OA_ALL) * 100, 2)}' \
                f'\nAA={round(np.mean(AA_ALL) * 100, 2)} +- {round(np.std(AA_ALL) * 100, 2)}' \
                f'\nKpp={round(np.mean(KPP_ALL) * 100, 2)} +- {round(np.std(KPP_ALL) * 100, 2)}')
    
    del net