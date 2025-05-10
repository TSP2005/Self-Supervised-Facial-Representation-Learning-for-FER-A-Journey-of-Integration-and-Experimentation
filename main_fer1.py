#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Facial Expression Recognition (FER)
https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013
https://paperswithcode.com/sota/facial-expression-recognition-on-fer-1
https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db
https://github.com/LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch
https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
https://www.kaggle.com/code/balmukund/fer-2013-pytorch-implementation
"""

__author__ = "GZ"

import os
import sys
import time
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from classy_vision.generic.distributed_util import is_distributed_training_run

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from data.fer2013 import FER2013, FERplus
from data.rafdb import RAFDB
from data.base_dataset import ImageFolderInstance
import data.transforms as data_transforms
from data.transforms import AddGaussianNoise
from data.sampler import DistributedImbalancedSampler, DistributedSamplerWrapper, ImbalancedDatasetSampler
import backbone as customized_models
from models import farl
from engine import validate
from utils import utils, get_norm
from utils.dist_utils import init_distributed_mode, all_reduce_mean

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='in1k',
                    help='name of dataset')
parser.add_argument('--data-root', default="",
                    help='root of dataset folder')
parser.add_argument('--trainindex', default=None, type=str, metavar='PATH',
                    help='path to train annotation (default: None)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--cls', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer', default="sgd", type=str, help='[sgd, adamw]')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument("--lr_head", default=0.02, type=float, help="initial learning rate - head")
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('--scheduler', default="cos", type=str, help='[cos, step, exp]')
parser.add_argument('--decay_epochs', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument("--gamma", type=float, default=0.1, help="lr decay factor")

parser.add_argument('--save-dir', default="ckpts",
                    help='checkpoint directory')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# dist
parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
    distributed training; """)
parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to self-supervised pretrained checkpoint')
parser.add_argument('--model-prefix', default='encoder_q', type=str,
                    help='the model prefix of self-supervised pretrained state_dict')
parser.add_argument('--norm', default='None', type=str,
                    help='the normalization for backbone (default: None)')
parser.add_argument('--amp', action='store_true', help='use automatic mixed precision training')

parser.add_argument('--train-percent', default=1.0, type=float, help='percentage of training set')
parser.add_argument('--model_type', default="ours", type=str, help='type of model')
parser.add_argument('--image_size', type=int, default=224, help='image size')
parser.add_argument('--data_norm', default="vggface2", type=str, help='type of data/transformation norm')
parser.add_argument('--finetune', action='store_true', help='fine-tune downstream backbone')

NUM_CLASSES = {'rafdb': 7, "fer2013": 7, "ferplus": 8, "affectnet7": 7}
best_acc1 = 0


def main(args):
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.model_type == "ours":
        norm = get_norm(args.norm)
        model = models.__dict__[args.arch](num_classes=NUM_CLASSES[args.dataset], norm_layer=norm)
    elif args.model_type == "farl":
        model_path = "./checkpoints/FaRL-Base-Patch16-LAIONFace20M-ep16.pth"
        vt = farl.FaRLVisualFeatures(model_type="base", model_path=model_path, forced_input_resolution=224)
        model = farl.FaRLClsWrapper(vt, num_classes=NUM_CLASSES[args.dataset])
    print(model)

    if not args.finetune:
        # Freeze all layers except layer3, layer4, and fc
        for name, param in model.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict'] if "state_dict" in checkpoint else checkpoint
            
            # Flexible key adjustment
            adjusted_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if args.model_prefix and new_key.startswith(args.model_prefix + "."):
                    new_key = new_key.replace(args.model_prefix + ".", "")
                elif new_key.startswith("module.encoder_q."):
                    new_key = new_key.replace("module.encoder_q.", "")
                elif new_key.startswith("module."):
                    new_key = new_key.replace("module.", "")
                adjusted_state_dict[new_key] = v

            args.start_epoch = 0
            msg = model.load_state_dict(adjusted_state_dict, strict=False)
            if len(msg.missing_keys) > 0:
                print("Missing keys:\n{}".format('\n'.join(msg.missing_keys)))
            if len(msg.unexpected_keys) > 0:
                print("Unexpected keys:\n{}".format('\n'.join(msg.unexpected_keys)))
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.pretrained, checkpoint.get('epoch', 'N/A')))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.multiprocessing_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        args.batch_size = int(args.batch_size / args.world_size)
        args.workers = int((args.workers + args.world_size - 1) / args.world_size)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # set optimizer
    if not args.finetune:
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(params) > 2  # More than just fc.weight and fc.bias
    else:
        trunk_parameters = []
        head_parameters, head_names = [], []
        for name, param in model.named_parameters():
            if name.startswith('fc') or name.startswith('module.fc'):
                head_parameters.append(param)
                head_names.append(name)
            else:
                trunk_parameters.append(param)
        assert len(head_parameters) == 2
        print("classifier params: {} using lr {}".format(head_names, args.lr_head))
        print("the rest params using lr {}".format(args.lr))
        params = [{'params': trunk_parameters},
                  {'params': head_parameters, 'lr': args.lr_head}]

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None and isinstance(best_acc1, torch.Tensor):
                best_acc1 = best_acc1.to(args.gpu)
            
            # Load model state
            state_dict = checkpoint['state_dict'] if "state_dict" in checkpoint else checkpoint
            adjusted_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if args.model_prefix and new_key.startswith(args.model_prefix + "."):
                    new_key = new_key.replace(args.model_prefix + ".", "")
                elif new_key.startswith("module.encoder_q."):
                    new_key = new_key.replace("module.encoder_q.", "")
                elif new_key.startswith("module."):
                    new_key = new_key.replace("module.", "")
                adjusted_state_dict[new_key] = v

            msg = model.load_state_dict(adjusted_state_dict, strict=False)
            if len(msg.missing_keys) > 0:
                print("Missing keys:\n{}".format('\n'.join(msg.missing_keys)))
            if len(msg.unexpected_keys) > 0:
                print("Unexpected keys:\n{}".format('\n'.join(msg.unexpected_keys)))
            
            # Try loading optimizer state, fallback to reinitialization if it fails
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded optimizer state from checkpoint")
            except ValueError as e:
                print(f"Warning: Optimizer state mismatch - {e}. Reinitializing optimizer.")
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=data_transforms.IMG_MEAN[args.data_norm],
                                     std=data_transforms.IMG_STD[args.data_norm])
    transform_train = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.5)
    ])
    transform_test = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset.lower() == "fer2013":
        train_dataset = FER2013(root=args.data_root, split="Training", transform=transform_train, convert_rgb=True)
        val_dataset = FER2013(root=args.data_root, split="PrivateTest", transform=transform_test, convert_rgb=True)
    elif args.dataset.lower() == "ferplus":
        train_dataset = FERplus(root=args.data_root, split="Training", transform=transform_train, convert_rgb=True)
        val_dataset = FERplus(root=args.data_root, split="PrivateTest", transform=transform_test, convert_rgb=True)
    elif args.dataset.lower() == "rafdb":
        train_dataset = RAFDB(root=args.data_root, split="train", transform=transform_train)
        val_dataset = RAFDB(root=args.data_root, split="test", transform=transform_test)
    elif args.dataset.lower() == "affectnet7":
        train_dataset = ImageFolderInstance(os.path.join(args.data_root, "train"), transform=transform_train)
        val_dataset = ImageFolderInstance(os.path.join(args.data_root, "val"), transform=transform_test)

    if args.train_percent < 1.0:
        num_subset = int(len(train_dataset) * args.train_percent)
        indices = torch.randperm(len(train_dataset))[:num_subset]
        indices = indices.tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print("Sub train_dataset:\n{}".format(len(train_dataset)))

    print(train_dataset)
    print(transform_train)

    train_sampler = None
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    if args.dataset.lower() == "affectnet7":
        if args.multiprocessing_distributed:
            train_sampler = DistributedSamplerWrapper(ImbalancedDatasetSampler(train_dataset))
        else:
            train_sampler = ImbalancedDatasetSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        if args.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scaler, epoch, args)

        if (epoch + 1) % args.eval_freq == 0:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_epoch = epoch

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % args.world_size == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best,
                    dir=args.save_dir,
                    filename='fer.pth.tar'
                )
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % args.world_size == 0):
            if epoch == args.start_epoch and not args.finetune:
                sanity_check(model.state_dict(), args.pretrained, args)

    print('Best Acc@1 {0} @ epoch {1}'.format(best_acc1, best_epoch + 1))


def train(train_loader, model, criterion, optimizer, scaler, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    lr_trunk = optimizer.param_groups[0]['lr']
    if args.finetune:
        lr_head = optimizer.param_groups[1]["lr"]
    else:
        lr_head = lr_trunk
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}/{}]\t"
               "LR trunk/head: {:.7f}/{:.7f}\t".format(epoch, args.epochs, lr_trunk, lr_head))

    if args.finetune:
        model.train()
    else:
        model.eval()

    end = time.time()
    for i, (images, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if scaler is None:
            output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        if is_distributed_training_run():
            acc1 = all_reduce_mean(acc1)
            acc5 = all_reduce_mean(acc5)

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, dir, filename='checkpoint.pth.tar'):
    os.makedirs(dir, exist_ok=True)
    file_path = os.path.join(dir, filename)
    torch.save(state, file_path)


def sanity_check(state_dict, pretrained_weights, args):
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    model_prefix = 'module.' + args.model_prefix + '.'
    for k in list(state_dict.keys()):
        if 'fc.weight' in k or 'fc.bias' in k:
            continue
        k_pre = model_prefix + k[len('module.'):] if k.startswith('module.') else model_prefix + k
        k_pre = k_pre.replace('backbone.fc.', 'neck.mlp.')
        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)
    print("=> sanity check passed.")


def adjust_learning_rate(optimizer, epoch, args):
    num_groups = len(optimizer.param_groups)
    assert 1 <= num_groups <= 2
    lrs = []
    if num_groups == 1:
        lrs += [args.lr]
    elif num_groups == 2:
        lrs += [args.lr, args.lr_head]
    assert len(lrs) == num_groups
    for group_id, param_group in enumerate(optimizer.param_groups):
        lr = lrs[group_id]
        if args.scheduler == "cos":
            if epoch < args.warmup_epochs:
                lr = lr * epoch / args.warmup_epochs
            else:
                lr = args.min_lr + (lr - args.min_lr) * 0.5 * \
                     (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif args.scheduler == "step":
            for milestone in args.decay_epochs:
                lr *= 0.1 if epoch >= milestone else 1.
        param_group['lr'] = lr


if __name__ == '__main__':
    print("Raw arguments:", sys.argv)
    # Use parse_known_args to handle distributed launch arguments gracefully
    opt, unknown = parser.parse_known_args()
    print("Parsed arguments:", opt)
    print("Unknown arguments (ignored):", unknown)
    init_distributed_mode(opt)
    main(opt)