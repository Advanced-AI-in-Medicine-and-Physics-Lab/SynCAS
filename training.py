from typing import Tuple, Union, List
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
import numpy as np
from batchgeneratorsv2.transforms.utils.random import RandomTransform
import os, sys
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from dataloading.dataset import defaultDataset
from dataloading.data_loader_3d import DataLoader3D
from dataloading.utils import get_allowed_n_proc_DA, Logger, dummy_context,empty_cache,collate_outputs
import torch
from torch.cuda.amp import GradScaler
from datetime import datetime
from time import time, sleep
from loss.losse import DC_and_CE_loss,PolyLRScheduler,DC_and_CE_lossWeight
from torch import autocast, nn
from torch import distributed as dist
from loss.dice import get_tp_fp_fn_tn
# from torch._dynamo import OptimizedModule
# from models.unet3d import UNet3D
from predictor import Predictor,export_prediction_from_logits,compute_metrics_on_folder
import warnings,multiprocessing
import random
import csv
from models.network import Unet,ProjectionHead
import torch.nn.functional as F
from test_from_file import InferencePreprocessor


class trainingPlanner(object):
    def __init__(self, plans: dict, fold: int,
                 device: torch.device = torch.device('cuda')):
        
        self.is_parallel = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_parallel else dist.get_rank()

        self.device = device
        self.fold = fold
        self.plans = plans
        self.alpha = 0.999
        self.beta = 0.03
        self.tau = 300
        self.test_latest_every = self.plans['test_every']
        self._best_dice = None

        self.ignore_label = None  #None

        self.preprocessed_dataset_folder_base = join(self.plans['data_preprocessed'], self.plans['dataset_name'])

        self.output_folder_base = join(self.plans['exp_results'], self.plans['dataset_name'],
                                       self.__class__.__name__ + '__' + self.plans['plans_name']) 
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.plans['data_identifier'])
        
        self.orininal_data_folder = None

        ### Some hyperparameters for you to fiddle with
        ### Some hyperparameters for you to fiddle with
        self.Augmented = bool(self.plans['augmentated'])


        self.batch_size = int(self.plans['batch_size'])

        self.labeled_batch = int(self.plans['batch_size'])

        self.initial_lr = float(self.plans['initial_lr'])  
        self.weight_decay = float(self.plans['weight_decay'])  
        self.oversample_foreground_percent = float(self.plans['oversample_foreground_percent'])  
        self.num_iterations_per_epoch = int(self.plans['num_iterations_per_epoch']) 
        self.num_val_iterations_per_epoch = int(self.plans['num_val_iterations_per_epoch'])  
        self.num_epochs = int(self.plans['num_epochs'])  

        self.current_epoch = 0
        self.enable_deep_supervision = False

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self.build_network_architecture()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize

        self.normal_weight = 0

        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = Logger()

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        self.inference_allowed_mirroring_axes = (0,1,2)  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 1
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers

        self.was_initialized = False

        self.print_to_log_file("\n#######################################################################\n"
                               "Trainer has been bulit."
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)

        self.inferencer = InferencePreprocessor(self.plans)
        
    def initialize(self):
        if not self.was_initialized:

            empty_cache(self.device)

            self.num_input_channels = 1

            self.is_parallel = False

            self.network = Unet(dimension=3, input_nc=self.num_input_channels, output_nc=len(self.plans['annotated_classes_key']), num_downs=4, ngf=16).to(self.device)

           

            decoder_channels = [128, 64, 32, 16]  
            self.projection_heads = nn.ModuleList([
                ProjectionHead(in_dim=ch) for ch in decoder_channels
            ]).cuda()
            if self.is_parallel:
                self.network = nn.DataParallel(self.network, device_ids=[0, 1]).to(self.device)
                # self.projection_heads = nn.DataParallel(self.projection_heads, device_ids=[0, 1]).to(self.device)
                        #UNet3D(in_channels=1, num_classes=2).to(self.device)
                        #
            self.optimizer = torch.optim.SGD(list(self.network.parameters()) + list(self.projection_heads.parameters()),
                                              self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
            self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.num_epochs)
            
            #torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
            # if ddp, wrap in DDP wrapper
            # if self.is_parallel:
            #     self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            #     self.network = DDP(self.network, device_ids=[self.local_rank])
            # if self.plans['with_normal']:
            #     print('Training involving normal samples.')

           
            self.loss = DC_and_CE_loss({'batch_dice': False,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_parallel}, {}, weight_ce=1, weight_dice=1, ignore_label=self.ignore_label)
            
            self.was_initialized = True

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_parallel:
                    mod = self.network.module
                    pro = self.projection_heads
                else:
                    mod = self.network
                    pro = self.projection_heads
                if False:#isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'projection_heads_weights': pro.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.plans,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')



    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

   
    def my_transforms(self, patch_size: Union[np.ndarray, Tuple[int]], Aug=False) :
        transforms = []
        
        patch_size_spatial = patch_size
        rotation_for_DA = (-0.5235987755982988, 0.5235987755982988)
        ignore_axes = False
        mirror_axes = (0,1,2)

        if Aug:
            transforms.append(
                SpatialTransform(
                    patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                    p_rotation=0.2,
                    rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.85, 1.25), p_synchronize_scaling_across_axes=1,
                    bg_style_seg_sampling=False  # , mode_seg='nearest'
                )
            )
            transforms.append(RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.1),
                    p_per_channel=1,
                    synchronize_channels=True
                ), apply_probability=0.1
            ))
            transforms.append(RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5, benchmark=True
                ), apply_probability=0.2
            ))
            transforms.append(RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.75, 1.25)),
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.15
            ))
            transforms.append(RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.75, 1.25)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.15
            ))
            transforms.append(RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5
                ), apply_probability=0.25
            ))
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.1
            ))
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.3
            ))
            if mirror_axes is not None and len(mirror_axes) > 0:
                transforms.append(
                    MirrorTransform(
                        allowed_axes=mirror_axes
                    )
                )
        else:
            transforms.append(
                SpatialTransform(
                    patch_size_spatial, patch_center_dist_from_border=0, random_crop=False
                    )
            )
            
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        return ComposeTransforms(transforms)    
    
    
    def my_transforms_val(self) :
            transforms = []

            transforms.append(
                RemoveLabelTansform(-1, 0)
            )

            return ComposeTransforms(transforms)

    def get_tr_and_val_datasets(self):
        splits_file = os.path.join(self.preprocessed_dataset_folder_base, "splits_final.json")
        # normal_file = os.path.join(self.preprocessed_dataset_folder_base, "normal.json")
        splits = load_json(splits_file)
        tr_keys_all = splits[self.fold]['train']
        val_keys = splits[self.fold]['val']

        # normal = load_json(normal_file)
        # normal_keys = normal[0]['train']

        tr_keys = tr_keys_all

        tr_keys = tr_keys
        # random.shuffle(normal_keys)

        dataset_tr = defaultDataset(self.preprocessed_dataset_folder, tr_keys)
        dataset_val = defaultDataset(self.preprocessed_dataset_folder, val_keys)
        # dataset_tr_normal = defaultDataset(self.preprocessed_dataset_folder, normal_keys)

        return dataset_tr, dataset_val#, dataset_tr_normal

    def get_dataloaders(self):
        patch_size = self.plans['patch_size']
        dim = len(patch_size)
        initial_patch_size = self.plans['initial_patch_size']
       
        # training pipeline
        tr_transforms = self.my_transforms(patch_size, self.Augmented)

        # validation pipeline
        val_transforms = self.my_transforms_val()

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()


        dl_tr = DataLoader3D(dataset_tr, self.batch_size,
                                    initial_patch_size,
                                    patch_size,
                                    self.plans,
                                    sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
        dl_val = DataLoader3D(dataset_val, self.batch_size,
                                    patch_size,
                                    patch_size,
                                    self.plans,
                                    sampling_probabilities=None, pad_sides=None, transforms=val_transforms)
        # dl_tr_normal = DataLoader3D(dataset_tr_normal, 1,
        #                             initial_patch_size,
        #                             patch_size,
        #                             self.plans,
        #                             sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
            # mt_gen_train_normal = SingleThreadedAugmenter(dl_tr_normal, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                        transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                        num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                        pin_memory=self.device.type == 'cuda',
                                                        wait_time=0.002)
            # mt_gen_train_normal = NonDetMultiThreadedAugmenter(data_loader=dl_tr_normal, transform=None,
            #                                             num_processes=allowed_num_processes,
            #                                             num_cached=max(6, allowed_num_processes // 2), seeds=None,
            #                                             pin_memory=self.device.type == 'cuda', wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        # _ = next(mt_gen_train_normal)
        return mt_gen_train, mt_gen_val#, mt_gen_train_normal


    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']


        # w_sample = (1 - self.alpha * torch.exp(torch.tensor(-self.beta * self.current_epoch))) * (1 - label_cal) + label_cal

        # w_sample = w_sample.to(self.device, non_blocking=True)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        image1 = data[:,0,:,:,:].unsqueeze(1)
        # image2 = data[:,1,:,:,:].unsqueeze(1)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output,_ = self.network(image1)
            del data
            l = self.loss(output, target)

       

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        # if self.label_manager.has_regions:
        #     predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        # else:
        # no need for softmax
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

     
        mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        # if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def contrastive_loss(self, feat1, feat2, label, temperature=0.1, num_samples=256, neg_per_anchor=2):
        """
        Efficient contrastive loss with per-anchor negatives (vectorized version).
        """

        if self.current_epoch > self.plans['num_epochs']/2:
            neg_per_anchor = neg_per_anchor*3
        B, C, D, H, W = feat1.shape

        # Flatten spatial dims: [B, N, C], where N = D*H*W
        feat1_flat = feat1.view(B, C, -1).transpose(1, 2)  # [B, N, C]
        feat2_flat = feat2.view(B, C, -1).transpose(1, 2)  # [B, N, C]

        # Downsample label to feature map resolution
        label_down = F.interpolate(label.float(), size=(D, H, W), mode="nearest")  # [B, 1, D, H, W]
        label_flat = label_down.view(B, -1)  # [B, N]

        total_loss = 0
        valid_batches = 0

        for b in range(B):
            pos_indices = torch.where(label_flat[b] > 0)[0]
            neg_indices = torch.where(label_flat[b] == 0)[0]

            if len(pos_indices) < 10 or len(neg_indices) < 10:
                continue  # skip if not enough samples

            # Sample anchor positions from positive indices
            if len(pos_indices) >= num_samples:
                anchor_ids = pos_indices[torch.randperm(len(pos_indices))[:num_samples]]
            else:
                anchor_ids = pos_indices[torch.randint(0, len(pos_indices), (num_samples,))]

            # For each anchor, sample neg_per_anchor negatives: vectorized sampling (idx: [num_samples, neg_per_anchor])
            if len(neg_indices) >= neg_per_anchor:
                rand_idx = torch.randint(0, len(neg_indices), (num_samples, neg_per_anchor), device=feat1.device)
            else:
                rand_idx = torch.randint(0, len(neg_indices), (num_samples, neg_per_anchor), device=feat1.device)
            neg_sample_ids = neg_indices[rand_idx]  # shape: [num_samples, neg_per_anchor]

            # Gather features
            anchor_feats = feat1_flat[b, anchor_ids]      # [num_samples, C]
            pos_feats = feat2_flat[b, anchor_ids]         # [num_samples, C]
            neg_feats = feat2_flat[b, neg_sample_ids]     # [num_samples, neg_per_anchor, C]

            # Compute logits: anchor_feats [N, C] × candidates.T → logits [N, 1+neg_per_anchor]
            # Step 1: positive logits (anchor * pos_feats): [N, 1]
            pos_logits = torch.sum(anchor_feats * pos_feats, dim=1, keepdim=True) / temperature  # [N, 1]

            # Step 2: negative logits (batch-wise matmul): anchor_feats [N, C] vs neg_feats [N, neg_per_anchor, C]
            neg_logits = torch.bmm(neg_feats, anchor_feats.unsqueeze(2)).squeeze(2) / temperature  # [N, neg_per_anchor]

            # Combine logits: [N, 1 + neg_per_anchor]
            logits = torch.cat([pos_logits, neg_logits], dim=1)  # [N, 1+neg_per_anchor]

            # Labels: positive is always index 0
            labels = torch.zeros(num_samples, dtype=torch.long, device=feat1.device)

            # Compute cross entropy loss
            loss = F.cross_entropy(logits, labels)

            total_loss += loss
            valid_batches += 1

        if valid_batches == 0:
            return torch.tensor(0.0, device=feat1.device)

        return total_loss / valid_batches

    def contrastive_loss_branchAware(self, feat1, feat2, label, temperature=0.3, num_samples=256, 
                                     base_neg_per_anchor=5, 
                                     base_pseudo_neg_per_anchor=2,     
                                     base_pos_per_anchor=1,alpha=0.5):
        """
        Contrastive loss with real negatives (label==0), positives (label==1),
        and pseudo-negatives (label>1), with distance scaling.

        The larger the alpha, the more influence is, =0, total neglect, =1 equal to negative.
        """

        pseudo_neg_per_anchor = base_pseudo_neg_per_anchor
        if self.current_epoch > self.plans['num_epochs']/2:
            base_neg_per_anchor *= 2
            pseudo_neg_per_anchor *= 2

        B, C, D, H, W = feat1.shape

        # Flatten spatial dims: [B, N, C]
        feat1_flat = feat1.view(B, C, -1).transpose(1, 2)  # [B, N, C]
        feat2_flat = feat2.view(B, C, -1).transpose(1, 2)  # [B, N, C]

        # Downsample label
        label_down = F.interpolate(label.float(), size=(D, H, W), mode="nearest")  # [B, 1, D, H, W]
        label_flat = label_down.view(B, -1)  # [B, N]

        total_loss = 0
        valid_batches = 0

        for b in range(B):
            label_b = label_flat[b]
            pos_indices = torch.where(label_b == 1)[0]
            neg_indices = torch.where(label_b == 0)[0]
            pseudo_neg_indices = torch.where(label_b > 1)[0]

            if len(pos_indices) < 10 or len(neg_indices) < 10:
                continue

            # Sample anchors from positive voxels
            if len(pos_indices) >= num_samples:
                anchor_ids = pos_indices[torch.randperm(len(pos_indices))[:num_samples]]
            else:
                anchor_ids = pos_indices[torch.randint(0, len(pos_indices), (num_samples,), device=feat1.device)]

            # Sample real negatives
            neg_rand_idx = torch.randint(0, len(neg_indices), (num_samples, base_neg_per_anchor), device=feat1.device)
            neg_sample_ids = neg_indices[neg_rand_idx]  # [num_samples, neg_per_anchor]

            # Sample pseudo-negatives if available
            if len(pseudo_neg_indices) > 0:
                pseudo_rand_idx = torch.randint(0, len(pseudo_neg_indices), (num_samples, pseudo_neg_per_anchor), device=feat1.device)
                pseudo_sample_ids = pseudo_neg_indices[pseudo_rand_idx]  # [num_samples, pseudo_neg_per_anchor]
            else:
                pseudo_sample_ids = None

            # Gather features
            anchor_feats = feat1_flat[b, anchor_ids]       # [N, C]
            pos_feats = feat2_flat[b, anchor_ids]          # [N, C]
            neg_feats = feat2_flat[b, neg_sample_ids]      # [N, neg_per_anchor, C]

            # Step 1: Positive logits
            pos_logits = torch.sum(anchor_feats * pos_feats, dim=1, keepdim=True) / temperature  # [N, 1]

            # Step 2: Negative logits
            neg_logits = torch.bmm(neg_feats, anchor_feats.unsqueeze(2)).squeeze(2) / temperature  # [N, neg_per_anchor]

            # Step 3: Pseudo-negative logits (optional)
            if pseudo_sample_ids is not None:
                pseudo_feats = feat2_flat[b, pseudo_sample_ids]  # [N, pseudo_neg_per_anchor, C]
                pseudo_logits = torch.bmm(pseudo_feats, anchor_feats.unsqueeze(2)).squeeze(2) / temperature  # [N, pseudo_neg_per_anchor]
                # Optionally scale down pseudo-negative logits (we treat them as weaker negatives)
                pseudo_logits = pseudo_logits * alpha  # can be tuned
                logits = torch.cat([pos_logits, neg_logits, pseudo_logits], dim=1)  # [N, 1 + neg + pseudo_neg]
            else:
                logits = torch.cat([pos_logits, neg_logits], dim=1)  # [N, 1 + neg]

            # Labels: index 0 is positive
            labels = torch.zeros(num_samples, dtype=torch.long, device=feat1.device)

            # Compute cross entropy loss
            loss = F.cross_entropy(logits, labels)

            total_loss += loss
            valid_batches += 1

        if valid_batches == 0:
            return torch.tensor(0.0, device=feat1.device)

        return total_loss / valid_batches


    def train_step(self, batch: dict) -> dict:

        data = batch['data']
        target = batch['target']
        branch_label = batch['branch_label']


        image1 = data[:,0,:,:,:].unsqueeze(1)
        image2 = data[:,1,:,:,:].unsqueeze(1)

        image1 = image1.to(self.device, non_blocking=True)
        image2 = image2.to(self.device, non_blocking=True)
        # print(w_sample.shape, data.shape)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        branch_label = branch_label.to(self.device, non_blocking=True)
        

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Forward pass: capture intermediate features
            # with torch.no_grad():
            pred1, feats1 = self.network(image1,CL=True)
            pred2, feats2 = self.network(image2,CL=True)

            loss_dice = (self.loss(pred1, target)+self.loss(pred2, target))/2
            # Contrastive loss
            loss_contrast = 0
            for idx in range(len(feats1)):
                proj_1 = self.projection_heads[idx](feats1[idx])
                proj_2 = self.projection_heads[idx](feats2[idx])
                if self.plans['branch_aware']:
                    # print('Using branch-aware contrastive loss.')
                    loss_contrast += self.contrastive_loss_branchAware(proj_1, proj_2, branch_label, 
                                                                       num_samples=self.plans['num_anchor'], 
                                                                       base_neg_per_anchor=self.plans['neg_per_anchor'],
                                                                       base_pos_per_anchor= self.plans['pos_per_anchor'], 
                                                                       base_pseudo_neg_per_anchor=self.plans['pseudo_neg_per_anchor'])
                else:
                    loss_contrast += self.contrastive_loss(proj_1, proj_2, target, num_samples=self.plans['num_anchor'], base_neg_per_anchor=self.plans['neg_per_anchor'])


            contrast_weight = 0.1
            # print('loss_dice:', loss_dice.item(), 'loss_contrast:', loss_contrast.item())
            l = loss_dice + contrast_weight * loss_contrast

            # l.backward()
            # self.optimizer.step()

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    
    def test_on_epoch(self, epoch: int):
        self.inferencer.network = self.network
        save_folder = join(self.output_folder, 'validation_external', str(epoch))
        self.inferencer.predict_from_files(join(self.plans["evaluation_folder"],'Images'), save_folder)

        dice = self.inferencer.calculate_Matrics_from_folder({
            'original_data_folder': join(self.plans["evaluation_folder"],'Images'),
            'gt_segmentations_folder': join(self.plans["evaluation_folder"],'GT'),
            'validation_output_folder': save_folder
         })

        return dice
    
    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_parallel:
            if False:#isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if False:#isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
    
    def perform_actual_validation(self, save_probabilities: bool = False):
        self.network.eval()

        if self.is_parallel and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = Predictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans,  None,
                                        self.inference_allowed_mirroring_axes)
        
        default_num_processes = 8

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            splits_file = os.path.join(self.preprocessed_dataset_folder_base, "splits_final.json")
            splits = load_json(splits_file)
            val_keys = splits[self.fold]['val']

            if self.is_parallel:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

            dataset_val = defaultDataset(self.preprocessed_dataset_folder, val_keys)

            results = []

            for i, k in enumerate(dataset_val.keys()):

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                # this needs to go into background processes
                
                export_prediction_from_logits(prediction, properties, self.plans,
                        output_filename_truncated, save_probabilities)

                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_parallel and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_parallel:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(self.orininal_data_folder, join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                '.nii.gz',
                                                self.plans['foreground_labels'],
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_parallel else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)
            self.print_to_log_file("Mean Validation HD95: ", (metrics['foreground_mean']["HD95"]),
                                   also_print_to_console=True)
            
            print("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]))
            print("Mean Validation HD95: ", (metrics['foreground_mean']["HD95"]))

    def calculate_Matrics_from_folder(self):
        validation_output_folder = join(self.output_folder, 'validation')
        default_num_processes = 8
        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(self.orininal_data_folder, join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                '.nii.gz',
                                                self.plans['foreground_labels'],
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_parallel else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)
            self.print_to_log_file("Mean Validation HD95: ", (metrics['foreground_mean']["HD95"]),
                                   also_print_to_console=True)
            self.print_to_log_file("Mean Validation MSD: ", (metrics['foreground_mean']["MSD"]),
                                   also_print_to_console=True)
            
            # print('Dice, HD95, MSD, ScoreRef, ScorePred:', results['metrics'][r]['Dice'], results['metrics'][r]['HD95'], results['metrics'][r]['MSD'], results['metrics'][r]['Agatston_Ref'],results['metrics'][r]['Agatston_Pred'])
            self.save_results_to_csv(metrics['metric_per_case'], join(self.output_folder,'score.csv'))

            print("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]))
            print("Mean Validation HD95: ", (metrics['foreground_mean']["HD95"]))  
            print("Mean Validation MSD: ", (metrics['foreground_mean']["MSD"]))

    def save_results_to_csv(self, results, output_file):
        headers = ['reference_file', 'Agatston_Ref', 'Agatston_Pred']

        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow(headers)
            
            for item in results:
                reference_file = item['reference_file']
                agatston_ref = item['metrics'][1]['Agatston_Ref']
                agatston_pred = item['metrics'][1]['Agatston_Pred']
                
                writer.writerow([os.path.basename(reference_file), agatston_ref, agatston_pred])

    def start_train(self):
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        print(self.plans)

        save_json(self.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)


        for epoch in range(self.current_epoch, self.num_epochs):
            self.network.train()
            self.lr_scheduler.step(self.current_epoch)

            self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
            self.print_to_log_file('')
            self.print_to_log_file(f'Epoch {self.current_epoch}')
            self.print_to_log_file(
                f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
            
       
            
            # lrs are the same for all workers so we don't need to gather them in case of DDP training
            self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            outputs = collate_outputs(train_outputs)
            loss_here = np.mean(outputs['loss'])

            self.logger.log('train_losses', loss_here, self.current_epoch)

            self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

            self.print_to_log_file(
                f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

            # handling periodic checkpointing
            current_epoch = self.current_epoch
            if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
                self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

            #evalation
            with torch.no_grad():
                self.network.eval()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                outputs_val = collate_outputs(val_outputs)
                tp = np.sum(outputs_val['tp_hard'], 0)
                fp = np.sum(outputs_val['fp_hard'], 0)
                fn = np.sum(outputs_val['fn_hard'], 0)
                loss_here = np.mean(outputs_val['loss'])

                global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
                mean_fg_dice = np.nanmean(global_dc_per_class)
                self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
                self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
                self.logger.log('val_losses', loss_here, self.current_epoch)
                

            self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
            self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
            self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                                self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
            

            # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
            if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
                self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
                self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best_Pseudo.pth'))

            if self.current_epoch % self.test_latest_every == 0:
                with torch.no_grad():
                    self.network.eval()
                    dice = self.test_on_epoch(self.current_epoch)
                    # self.logger.log('val_dice', dice, self.current_epoch)
                    self.print_to_log_file('val_dice', np.round(dice, decimals=4))
                    
                    if self._best_dice is None or dice > self._best_dice:
                        self._best_dice = dice
                        self.print_to_log_file(f"Yayy! New best Dice: {np.round(dice, decimals=4)}")
                        self.save_checkpoint(join(self.output_folder, f'checkpoint_best_DICE_{self.current_epoch}.pth'))


            if self.local_rank == 0:
                self.logger.plot_progress_png(self.output_folder)

            self.current_epoch += 1

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")     


if __name__ =='__main__':
    # os.environ['TORCHDYNAMO_DISABLE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    

# def __init__(self, plans: dict, fold: int, dataset_json: dict,
#                  device: torch.device = torch.device('cuda')):

    plans = load_json('config.json')
    fold = 0
    myTrainer = trainingPlanner(plans, fold)

    myTrainer.load_checkpoint(join(myTrainer.output_folder, 'checkpoint_latest.pth'))
    myTrainer.start_train()
    # myTrainer.calculate_Matrics_from_folder()
    # myTrainer.load_checkpoint(join(myTrainer.output_folder, 'checkpoint_best.pth'))
    # myTrainer.perform_actual_validation(False)

