# Original copyright Amazon.com, Inc. or its affiliates, under CC-BY-NC-4.0 License.
# Modifications Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import time
from datetime import timedelta
import numpy as np
try:
    import faiss
except ImportError:
    pass

import torch
import torch.nn as nn
from classy_vision.generic.distributed_util import is_distributed_training_run

from utils import utils
from utils.dist_utils import all_reduce_mean
from sklearn.metrics import accuracy_score  # Added for per-class accuracy

def validate(val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if is_distributed_training_run():
                acc1 = all_reduce_mean(acc1)
                acc5 = all_reduce_mean(acc5)

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # Collect predictions and targets for per-class accuracy
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t().squeeze(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # Compute overall metrics
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.4f}'
              .format(top1=top1, top5=top5, loss=losses))

        # Compute per-class accuracy
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        classes = np.unique(all_targets)
        per_class_acc = {}
        for cls in classes:
            idx = all_targets == cls
            if idx.sum() > 0:
                per_class_acc[cls] = accuracy_score(all_targets[idx], all_preds[idx]) * 100
            else:
                per_class_acc[cls] = 0.0

        print('Per-class Acc@1:')
        for cls, acc in per_class_acc.items():
            print(f'Class {cls}: {acc:.2f}%')

    return top1.avg

def ss_validate(val_loader_base, val_loader_query, model, args):
    print("start KNN evaluation with key size={} and query size={}".format(
        len(val_loader_base.dataset.samples), len(val_loader_query.dataset.samples)))
    batch_time_key = utils.AverageMeter('Time', ':6.3f')
    batch_time_query = utils.AverageMeter('Time', ':6.3f')
    model.eval()

    feats_base, target_base, feats_query, target_query = [], [], [], []

    with torch.no_grad():
        start = time.time()
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader_base):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            feats = model(images)
            feats = nn.functional.normalize(feats, dim=1)
            feats_base.append(feats)
            target_base.append(target)

            batch_time_key.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(f'Extracting key features: [{i}/{len(val_loader_base)}]\t'
                      f'Time {batch_time_key.val:.3f} ({batch_time_key.avg:.3f})')

        end = time.time()
        for i, (images, target, _) in enumerate(val_loader_query):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            feats = model(images)
            feats = nn.functional.normalize(feats, dim=1)
            feats_query.append(feats)
            target_query.append(target)

            batch_time_query.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(f'Extracting query features: [{i}/{len(val_loader_query)}]\t'
                      f'Time {batch_time_query.val:.3f} ({batch_time_query.avg:.3f})')

        feats_base = torch.cat(feats_base, dim=0).detach().cpu().numpy()
        target_base = torch.cat(target_base, dim=0).detach().cpu().numpy()
        feats_query = torch.cat(feats_query, dim=0).detach().cpu().numpy()
        target_query = torch.cat(target_query, dim=0).detach().cpu().numpy()
        feat_time = time.time() - start

        index = faiss.IndexFlatL2(feats_base.shape[1])
        index.add(feats_base)
        D, I = index.search(feats_query, args.num_nn)
        preds = np.array([np.bincount(target_base[n]).argmax() for n in I])

        NN_acc = (preds == target_query).sum() / len(target_query) * 100.0
        knn_time = time.time() - start - feat_time
        print(f"finished KNN evaluation, feature time: {timedelta(seconds=feat_time)}, "
              f"knn time: {timedelta(seconds=knn_time)}")
        print(f' * NN Acc@1 {NN_acc:.3f}')

    return NN_acc

def ss_face_validate(val_loader, model, args, threshold=0.6):
    """
    Face verification validation using cosine similarity on CFP-FP pairs.
    https://github.com/sakshamjindal/Face-Matching
    """
    batch_time = utils.AverageMeter('Time', ':6.3f')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, top1],
        prefix='Test (CFP-FP): ')

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    model.eval()
    model = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        end = time.time()
        for i, (img1, img2, target) in enumerate(val_loader):
            img1 = img1.cuda(args.gpu, non_blocking=True)
            img2 = img2.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            embedding1, _, _ = model.online_net(img1)
            embedding2, _, _ = model.online_net(img2)

            embedding1 = embedding1.squeeze(-1)
            embedding2 = embedding2.squeeze(-1)

            assert embedding1.ndim == 2, f"Expected 2D embeddings, got {embedding1.ndim}"

            cosine_similarity = cos(embedding1, embedding2)
            pred = (cosine_similarity >= threshold).float()
            acc1 = (pred == target).float().sum() * 100.0 / target.shape[0]

            if is_distributed_training_run():
                acc1 = all_reduce_mean(acc1)

            top1.update(acc1.item(), img1.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(f' * Acc@1 {top1.avg:.3f}')

    return top1.avg

def validate_multilabel(val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True).float()

            output = model(images)
            loss = criterion(output, target)
            acc1 = utils.accuracy_multilabel(torch.sigmoid(output), target)

            if is_distributed_training_run():
                acc1 = all_reduce_mean(acc1)

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(f' * Acc@1 {top1.avg:.3f} Loss {losses.avg:.4f}')

    return top1.avg