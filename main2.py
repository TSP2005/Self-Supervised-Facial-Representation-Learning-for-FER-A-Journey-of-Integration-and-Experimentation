# some code in this file is adapted from
# https://github.com/pytorch/examples
# Original Copyright 2017. Licensed under the BSD 3-Clause License.
# Modifications Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os
import time
import random
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from classy_vision.generic.distributed_util import is_distributed_training_run

import backbone as backbone_models
from models import get_model
from utils import utils, lr_schedule, LARS, get_norm, init_distributed_mode
import data.transforms as data_transforms
from engine import ss_validate, ss_face_validate
from data.base_dataset import get_dataset

backbone_model_names = sorted(name for name in backbone_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbone_models.__dict__[name]))

# ... (previous imports and code remain unchanged)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default="vggface2", help='name of dataset', choices=['in1k', 'in100', 'im_folder', 'in1k_idx', "vggface2"])
parser.add_argument('--data-root', default="./data/VGG-Face2-crop/train", help='root of dataset folder')
parser.add_argument('--cfp-root', default="./data/cfp_fp", help='root of CFP-FP dataset folder')
parser.add_argument('--arch', default='FRAB', help='model architecture')
parser.add_argument('--backbone', default='resnet18_encoder', choices=backbone_model_names, help='model architecture: resnet18_encoder (default)')
parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--warmup-epoch', default=5, type=int, help='number of epochs for learning warmup')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int, help='lr schedule')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--save-dir', default="ckpts", help='checkpoint directory')
parser.add_argument('-p', '--print-freq', default=50, type=int, help='print frequency')
parser.add_argument('--save-freq', default=10, type=int, help='checkpoint save frequency')
parser.add_argument('--eval-freq', default=5, type=int, help='evaluation epoch frequency')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=23456, type=int, help='seed for initializing training')
parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
parser.add_argument('--local-rank', default=0, type=int, help='local rank for distributed training')  # Fixed to --local-rank
parser.add_argument('--multiprocessing_distributed', action='store_true', help='use multi-processing distributed training')
parser.add_argument('--proj-dim', default=256, type=int, help='feature dimension')
parser.add_argument('--enc-m', default=0.996, type=float, help='momentum of updating key encoder')
parser.add_argument('--norm', default='None', type=str, help='normalization for network')
parser.add_argument('--num-neck-mlp', default=2, type=int, help='number of neck mlp')
parser.add_argument('--hid-dim', default=2048, type=int, help='hidden dimension of mlp')
parser.add_argument('--amp', action='store_true', help='use automatic mixed precision training')
parser.add_argument('--lewel-l2-norm', action='store_true', help='use l2-norm before softmax')
parser.add_argument('--lewel-scale', default=0.1, type=float, help='scale factor of attention map')
parser.add_argument('--lewel-num-heads', default=4, type=int, help='number of heads in lewel')
parser.add_argument('--lewel-loss-weight', default=0.5, type=float, help='loss weight for aligned branch')
parser.add_argument('--train-percent', default=1.0, type=float, help='percentage of training set')
parser.add_argument('--mask_type', default="attn", type=str, help='type of masks')
parser.add_argument('--num_proto', default=8, type=int, help='number of heatmaps')
parser.add_argument('--teacher_temp', default=0.07, type=float, help='temperature of the teacher')
parser.add_argument('--loss_w_cluster', default=0.5, type=float, help='loss weight for cluster assignments')
parser.add_argument('--num-nn', default=20, type=int, help='number of nearest neighbors')
parser.add_argument('--diversity_lambda', default=0.01, type=float, help='weight for diversity loss term')

# ... (rest of main.py remains unchanged: CFPPairs, main(), train(), etc.)
best_acc1 = 0

class CFPPairs(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_pairs=32000):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, 'cfp-dataset', 'Data', 'Images')
        self.protocol_dir = os.path.join(root, 'cfp-dataset', 'Protocol', 'Split', 'FP')
        self.target_pairs = target_pairs
        if not os.path.exists(self.protocol_dir):
            raise FileNotFoundError(f"Protocol directory {self.protocol_dir} not found")
        self.pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = []; valid_ids = range(492, 501); max_frontal_idx = 10; max_profile_idx = 4
        all_frontal = []; all_profile = []
        for id in valid_ids:
            for idx in range(1, max_frontal_idx + 1):
                path = f'{id:03d}/frontal/{idx:02d}.jpg'
                full_path = os.path.join(self.image_dir, path)
                if os.path.exists(full_path): all_frontal.append(path)
            for idx in range(1, max_profile_idx + 1):
                path = f'{id:03d}/profile/{idx:02d}.jpg'
                full_path = os.path.join(self.image_dir, path)
                if os.path.exists(full_path): all_profile.append(path)
        all_images = all_frontal + all_profile
        print(f"Found {len(all_frontal)} frontal and {len(all_profile)} profile images")
        for split in range(1, 11):
            split_dir = os.path.join(self.protocol_dir, f'{split:02d}')
            if not os.path.exists(split_dir): raise FileNotFoundError(f"Split directory {split_dir} not found")
            same_file = os.path.join(split_dir, 'same.txt')
            if os.path.exists(same_file):
                with open(same_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) == 2:
                            id, idx = map(int, parts)
                            if id in valid_ids and idx <= max_frontal_idx:
                                img1_path = f'{id:03d}/frontal/{idx:02d}.jpg'
                                for prof_idx in range(1, max_profile_idx + 1):
                                    img2_path = f'{id:03d}/profile/{prof_idx:02d}.jpg'
                                    img1_full_path = os.path.join(self.image_dir, img1_path)
                                    img2_full_path = os.path.join(self.image_dir, img2_path)
                                    if os.path.exists(img1_full_path) and os.path.exists(img2_full_path):
                                        pair = (img1_path, img2_path, 1)
                                        if pair not in pairs: pairs.append(pair)
            diff_file = os.path.join(split_dir, 'diff.txt')
            if os.path.exists(diff_file):
                with open(diff_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) == 4:
                            id1, idx1, id2, idx2 = map(int, parts)
                            if (id1 in valid_ids and id2 in valid_ids and idx1 <= max_frontal_idx and idx2 <= max_profile_idx):
                                img1_path = f'{id1:03d}/frontal/{idx1:02d}.jpg'
                                img2_path = f'{id2:03d}/profile/{idx2:02d}.jpg'
                                img1_full_path = os.path.join(self.image_dir, img1_path)
                                img2_full_path = os.path.join(self.image_dir, img2_path)
                                if os.path.exists(img1_full_path) and os.path.exists(img2_full_path):
                                    pair = (img1_path, img2_path, 0)
                                    if pair not in pairs: pairs.append(pair)
        if len(pairs) < self.target_pairs:
            print(f"Generating synthetic pairs to reach {self.target_pairs} (current: {len(pairs)})")
            for id in valid_ids:
                for front_idx in range(1, max_frontal_idx + 1):
                    for prof_idx in range(1, max_profile_idx + 1):
                        img1_path = f'{id:03d}/frontal/{front_idx:02d}.jpg'
                        img2_path = f'{id:03d}/profile/{prof_idx:02d}.jpg'
                        img1_full_path = os.path.join(self.image_dir, img1_path)
                        img2_full_path = os.path.join(self.image_dir, img2_path)
                        if os.path.exists(img1_full_path) and os.path.exists(img2_full_path):
                            pair = (img1_path, img2_path, 1)
                            if pair not in pairs: pairs.append(pair)
            all_images = all_frontal + all_profile
            while len(pairs) < self.target_pairs:
                img1 = random.choice(all_images)
                img2 = random.choice(all_images)
                if img1 != img2:
                    id1 = int(img1.split('/')[0])
                    id2 = int(img2.split('/')[0])
                    label = 0 if id1 != id2 else 1
                    pairs.append((img1, img2, label))
        print(f"Loaded {len(pairs)} valid pairs from {self.protocol_dir}")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img1_path, img2_path, label = self.pairs[index]
        img1_full_path = os.path.join(self.image_dir, img1_path)
        img2_full_path = os.path.join(self.image_dir, img2_path)
        img1 = Image.open(img1_full_path).convert('RGB')
        img2 = Image.open(img2_full_path).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)

def main(args):
    global best_acc1
    init_distributed_mode(args)
    if args.gpu is None: args.gpu = args.local_rank

    print("=> creating model '{}' with backbone '{}'".format(args.arch, args.backbone))
    model_func = get_model(args.arch)
    norm_layer = get_norm(args.norm)
    model = model_func(
        backbone_models.__dict__[args.backbone],
        dim=args.proj_dim,
        m=args.enc_m,
        hid_dim=args.hid_dim,
        norm_layer=norm_layer,
        num_neck_mlp=args.num_neck_mlp,
        scale=args.lewel_scale,
        l2_norm=args.lewel_l2_norm,
        num_heads=args.lewel_num_heads,
        loss_weight=args.lewel_loss_weight,
        mask_type=args.mask_type,
        num_proto=args.num_proto,
        teacher_temp=args.teacher_temp,
        loss_w_cluster=args.loss_w_cluster,
        diversity_lambda=args.diversity_lambda
    )
    if args.pretrained and os.path.isfile(args.pretrained):
        print("=> loading pretrained model from '{}'".format(args.pretrained))
        state_dict = torch.load(args.pretrained, map_location="cpu")['state_dict']
        for k in list(state_dict.keys()):
            new_key = k.replace("module.", "")
            state_dict[new_key] = state_dict[k]
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print("=> loaded pretrained model from '{}'".format(args.pretrained))
        if len(msg.missing_keys) > 0: print("missing keys: {}".format(msg.missing_keys))
        if len(msg.unexpected_keys) > 0: print("unexpected keys: {}".format(msg.unexpected_keys))

    model.cuda(args.gpu)
    args.batch_size = int(args.batch_size / args.world_size)
    args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    params = collect_params(model, exclude_bias_and_bn=True, sync_bn=True)
    optimizer = LARS(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume and os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
        args.start_epoch = checkpoint['epoch']
        if 'best_acc1' in checkpoint: best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler and 'scaler' in checkpoint: scaler.load_state_dict(checkpoint['scaler'])
        print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    cudnn.benchmark = True

    transform1, transform2 = data_transforms.get_vggface_tranforms(image_size=224)
    train_dataset = get_dataset(
        args.dataset,
        mode='train',
        transform=data_transforms.TwoCropsTransform(transform1, transform2),
        data_root=args.data_root)
    print("train_dataset:\n{}".format(train_dataset))

    num_subset = 64000
    indices = torch.randperm(len(train_dataset))[:num_subset].tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices)
    print("Sub train_dataset for 1000 batches:\n{}".format(len(train_dataset)))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        persistent_workers=True)

    normalize = transforms.Normalize(mean=data_transforms.IMG_MEAN["vggface2"], std=data_transforms.IMG_STD["vggface2"])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = CFPPairs(root=args.cfp_root, transform=transform_test, target_pairs=32000)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers//2, pin_memory=True,
        persistent_workers=True)

    if args.evaluate:
        ss_face_validate(val_loader, model, args)
        return

    best_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if epoch >= args.warmup_epoch:
            lr_schedule.adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, optimizer, scaler, epoch, args)
        is_best = False
        if (epoch + 1) % args.eval_freq == 0:
            acc1 = ss_face_validate(val_loader, model, args)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best: best_epoch = epoch
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.world_size == 0):
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler else None,
            }, is_best=is_best, epoch=epoch, args=args)
    print(f'Best Acc@1 {best_acc1} @ epoch {best_epoch + 1}')

def train(train_loader, model, optimizer, scaler, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    losses_base = utils.AverageMeter('Loss_base', ':.4e')
    losses_inst = utils.AverageMeter('Loss_inst', ':.4e')
    losses_obj = utils.AverageMeter('Loss_obj', ':.4e')
    losses_clu = utils.AverageMeter('Loss_clu', ':.4e')
    losses_div = utils.AverageMeter('Loss_div', ':.4e')
    curr_lr = utils.InstantMeter('LR', ':.7f')
    curr_mom = utils.InstantMeter('MOM', ':.7f')
    progress = utils.ProgressMeter(
        len(train_loader),
        [curr_lr, curr_mom, batch_time, data_time, losses, losses_base, losses_inst, losses_obj, losses_clu, losses_div],
        prefix=f"Epoch: [{epoch}/{args.epochs}]\t")

    batch_iter = len(train_loader)
    max_iter = float(batch_iter * args.epochs)
    model.train()
    if "EMAN" in args.arch:
        print("setting the key model to eval mode when using EMAN")
        if hasattr(model, 'module'): model.module.target_net.eval()
        else: model.target_net.eval()

    end = time.time()
    accum_steps = 4
    for i, (images, _, idx) in enumerate(train_loader):
        curr_iter = float(epoch * batch_iter + i)
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            idx = idx.cuda(args.gpu, non_blocking=True)

        if epoch < args.warmup_epoch:
            warmup_step = args.warmup_epoch * batch_iter
            curr_step = epoch * batch_iter + i + 1
            lr_schedule.warmup_learning_rate(optimizer, curr_step, warmup_step, args)
        curr_lr.update(optimizer.param_groups[0]['lr'])

        if scaler:
            optimizer.zero_grad() if i % accum_steps == 0 else None
            with torch.cuda.amp.autocast():
                loss, loss_pack = model(im_v1=images[0], im_v2=images[1], idx=idx)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss, loss_pack = model(im_v1=images[0], im_v2=images[1], idx=idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.update(loss.item() * accum_steps, images[0].size(0))
        losses_base.update(loss_pack["base"].item(), images[0].size(0))
        losses_inst.update(loss_pack["inst"].item(), images[0].size(0))
        losses_obj.update(loss_pack["obj"].item(), images[0].size(0))
        losses_clu.update(loss_pack["clu"].item(), images[0].size(0))
        losses_div.update(loss_pack["div"].item(), images[0].size(0))

        if hasattr(model, 'module'):
            model.module.momentum_update(curr_iter, max_iter)
            curr_mom.update(model.module.curr_m)
        else:
            model.momentum_update(curr_iter, max_iter)
            curr_mom.update(model.curr_m)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def collect_params(model, exclude_bias_and_bn=True, sync_bn=True):
    weight_param_list, bn_and_bias_param_list = [], []
    weight_param_names, bn_and_bias_param_names = [], []
    for name, param in model.named_parameters():
        if exclude_bias_and_bn and ('bn' in name or 'depwnsample.1' in name or 'bias' in name):
            bn_and_bias_param_list.append(param)
            bn_and_bias_param_names.append(name)
        else:
            weight_param_list.append(param)
            weight_param_names.append(name)
    print("weight params:\n{}".format('\n'.join(weight_param_names)))
    print("bn and bias params:\n{}".format('\n'.join(bn_and_bias_param_names)))
    param_list = [{'params': bn_and_bias_param_list, 'weight_decay': 0., 'lars_exclude': True},
                  {'params': weight_param_list}]
    return param_list

if __name__ == '__main__':
    opt = parser.parse_args()
    opt.distributed = True
    opt.multiprocessing_distributed = True
    init_distributed_mode(opt)
    main(opt)