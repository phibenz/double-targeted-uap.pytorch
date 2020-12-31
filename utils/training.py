from __future__ import division
import numpy as np
import os, shutil, time
import itertools
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.utils import time_string, print_log


def adjust_learning_rate(init_lr, init_momentum, optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr
    momentum = init_momentum
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
            momentum = momentum * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # param_group['momentum'] = momentum
    return lr


def train_target_model(train_loader, model, criterion, optimizer, epoch, log,
            print_freq=200, use_cuda=True):
    # train function (forward, backward, update)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for iteration, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            target = target.cuda()
            input = input.cuda()

        # compute output
        if model.module._get_name() == "Inception3":
            output, aux_output = model(input)
            loss1 = criterion(output, target)
            loss2 = criterion(aux_output, target)
            loss = loss1 + 0.4*loss2
        else:
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=-1)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        epoch, iteration, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                    top5=top5,
                                                                                                    error1=100-top1.avg), log)
    return top1.avg, losses.avg


def train_half_half(sources_data_loader, others_data_loader,
                    model, target_model, criterion, optimizer, epsilon, num_iterations, log,
                    print_freq=200, use_cuda=True, patch=False):
    # train function (forward, backward, update)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.module.generator.train()
    model.module.target_model.eval()
    target_model.eval()

    end = time.time()

    sources_data_iterator = iter(sources_data_loader)
    others_data_iterator = iter(others_data_loader)

    iteration=0
    while (iteration<num_iterations):
        try:
            sources_input, sources_target = next(sources_data_iterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            sources_data_iterator = iter(sources_data_loader)
            sources_input, sources_target = next(sources_data_iterator)

        try:
            others_input, others_target = next(others_data_iterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            others_data_iterator = iter(others_data_loader)
            others_input, others_target = next(others_data_iterator)

        # Concat the two batches
        input = torch.cat([sources_input, others_input], dim=0)
        target = torch.cat([sources_target, others_target], dim=0)

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            target = target.cuda()
            input = input.cuda()

        # compute output
        output = model(input)
        target_model_output = target_model(input)
        loss = criterion(output, target_model_output, target)

        # measure accuracy and record loss
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=-1)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Project to l-infinity ball
        if patch:
            model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, 0, epsilon)
        else:
            model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, -epsilon, epsilon)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0:
            print_log('  Iteration: [{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        iteration, num_iterations, batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

        iteration+=1

    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                    top5=top5,
                                                                                                    error1=100-top1.avg), log)


def validate(val_loader, model, criterion, log=None, use_cuda=True):
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):

        if use_cuda:
            target = target.cuda()
            input = input.cuda()

        with torch.no_grad():
            # compute output
            output = model(input)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
    if log:
        print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                    top5=top5,
                                                                                                    error1=100-top1.avg), log)
    else:
        print('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg))

    return top1.avg


def metrics_evaluate(source_loader, others_loader, target_model, perturbed_model, source_classes, sink_classes, log=None, use_cuda=True):
    if len(sink_classes)!=0:
        assert len(source_classes) == len(sink_classes)

    # switch to evaluate mode
    target_model.eval()
    perturbed_model.eval()

    for loader, loader_name in zip([source_loader, others_loader],["Source", "Others"]):
        clean_acc = AverageMeter()
        perturbed_acc = AverageMeter()
        attack_success_rate = AverageMeter() # Among the correctly classified samples, the ratio of being different from clean prediction (same as gt)
        if len(sink_classes)!=0:
            all_to_sink_success_rate = []
            source_to_sink_success_rate = []
            all_to_sink_success_rate_filtered = []
            for i in range(len(sink_classes)):
                all_to_sink_success_rate.append(AverageMeter()) # The ratio of samples going to the sink classes
                source_to_sink_success_rate.append(AverageMeter())
                all_to_sink_success_rate_filtered.append(AverageMeter())

        total_num_samples = 0
        num_same_classified = 0
        num_diff_classified = 0

        if len(loader)>0: # For UAP, all classes will be attacked, so others_loader is empty
            for input, gt in loader:
                if use_cuda:
                    gt = gt.cuda()
                    input = input.cuda()

                # compute output
                with torch.no_grad():
                    clean_output = target_model(input)
                    pert_output = perturbed_model(input)

                correctly_classified_mask = torch.argmax(clean_output, dim=-1).cpu() == gt.cpu()
                cl_acc = accuracy(clean_output.data, gt, topk=(1,))
                clean_acc.update(cl_acc[0].item(), input.size(0))
                pert_acc = accuracy(pert_output.data, gt, topk=(1,))
                perturbed_acc.update(pert_acc[0].item(), input.size(0))

                # Calculating Fooling Ratio params
                clean_out_class = torch.argmax(clean_output, dim=-1)
                pert_out_class = torch.argmax(pert_output, dim=-1)

                total_num_samples += len(clean_out_class)
                num_same_classified += torch.sum(clean_out_class == pert_out_class).cpu().numpy()
                num_diff_classified += torch.sum(~(clean_out_class == pert_out_class)).cpu().numpy()

                if torch.sum(correctly_classified_mask)>0:
                    with torch.no_grad():
                        pert_output_corr_cl = perturbed_model(input[correctly_classified_mask])
                    attack_succ_rate = accuracy(pert_output_corr_cl, gt[correctly_classified_mask], topk=(1,))
                    attack_success_rate.update(attack_succ_rate[0].item(), pert_output_corr_cl.size(0))

                # Collect samples from Source go to sink
                if len(sink_classes)!=0:
                    # Iterate over source class and sink class pairs
                    for cl_idx, (source_cl, sink_cl) in enumerate(zip(source_classes, sink_classes)):

                        # 1. Check how many of the paired source class got to the sink class (Only relevant for source loader)
                        # Filter all idxs which belong to the source class
                        source_cl_idxs = [i == source_cl for i in gt]
                        source_cl_mask = torch.Tensor(source_cl_idxs)==True
                        if torch.sum(source_cl_mask)>0:
                            gt_source_cl = gt[source_cl_mask]
                            pert_output_source_cl = pert_output[source_cl_mask]

                            # Create desired target value
                            target_sink = torch.ones_like(gt_source_cl) * sink_cl
                            source_to_sink_succ_rate = accuracy(pert_output_source_cl, target_sink, topk=(1,))
                            source_to_sink_success_rate[cl_idx].update(source_to_sink_succ_rate[0].item(), pert_output_source_cl.size(0))

                        # 2. How many of all samples go the sink class (Only relevant for others loader)
                        target_sink = torch.ones_like(gt) * sink_cl
                        all_to_sink_succ_rate = accuracy(pert_output, target_sink, topk=(1,))
                        all_to_sink_success_rate[cl_idx].update(all_to_sink_succ_rate[0].item(), pert_output.size(0))

                        # 3. How many of all samples go the sink class, except gt sink class (Only relevant for others loader)
                        # Filter all idxs which are not belonging to sink class
                        non_sink_class_idxs = [i != sink_cl for i in gt]
                        non_sink_class_mask = torch.Tensor(non_sink_class_idxs)==True
                        if torch.sum(non_sink_class_mask)>0:
                            gt_non_sink_class = gt[non_sink_class_mask]
                            pert_output_non_sink_class = pert_output[non_sink_class_mask]

                            target_sink = torch.ones_like(gt_non_sink_class) * sink_cl
                            all_to_sink_succ_rate_filtered = accuracy(pert_output_non_sink_class, target_sink, topk=(1,))
                            all_to_sink_success_rate_filtered[cl_idx].update(all_to_sink_succ_rate_filtered[0].item(), pert_output_non_sink_class.size(0))

            if log:
                print_log('\n\t########## {} #############'.format(loader_name), log)
                if len(sink_classes)!=0:
                    for cl_idx, (source_cl, sink_cl) in enumerate(zip(source_classes, sink_classes)):
                        print_log('\n\tSource {} --> Sink {} Prec@1 {:.3f}'.format(source_cl, sink_cl, source_to_sink_success_rate[cl_idx].avg), log)
                        print_log('\tAll --> Sink {} Prec@1 {:.3f}'.format(sink_cl, all_to_sink_success_rate[cl_idx].avg), log)
                        # Average fooling ratio of the non-source classes into the target label
                        print_log('\tAll (w/o sink samples) --> Sink {} Prec@1 {:.3f}'.format(sink_cl, all_to_sink_success_rate_filtered[cl_idx].avg), log)
            else:
                print('\n\t########## {} #############'.format(loader_name))
                if len(sink_classes)!=0:
                    for cl_idx, (source_cl, sink_cl) in enumerate(zip(source_classes, sink_classes)):
                        print('\n\tSource {} --> Sink {} Prec@1 {:.3f}'.format(source_cl, sink_cl, source_to_sink_success_rate[cl_idx].avg))
                        print('\tAll --> Sink {} Prec@1 {:.3f}'.format(sink_cl, all_to_sink_success_rate[cl_idx].avg))
                        # Average fooling ratio of the non-source classes into the target label
                        print('\tAll (w/o sink samples) --> Sink {} Prec@1 {:.3f}'.format(sink_cl, all_to_sink_success_rate_filtered[cl_idx].avg))

def save_checkpoint(state, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()

  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 80
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)

    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)


    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)
