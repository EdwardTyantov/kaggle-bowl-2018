#-*- coding: utf8 -*-
import shutil, time, logging
import torch
import torch.optim
from utils import AverageMeter, dice_loss


logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)


class PlateauScheduler(object):
    """Sets the lr to the initial LR decayed by 1/decrease_rate, when not improving for max_stops epochs"""
    def __init__(self, optimizer, patience, early_stop_n, decrease_rate=0.1, eps=1e-5,
                 warm_up_epochs=None, best_score=None):
        self.optimizer = optimizer
        if not isinstance(optimizer, (torch.optim.SGD, )):
            raise TypeError
        self.patience = patience
        self.early_stop_n = early_stop_n
        self.decrease_rate = decrease_rate
        self.eps = eps
        self.warm_up_epochs = warm_up_epochs
        self.__lr_changed = 0
        self.__early_stop_counter = 0
        self.__best_score = best_score
        self.__descrease_times = 0
        self.__warm_up = None

    def step(self, epoch, score):
        adjusted, to_break = False, False

        prev_best_score = self.__best_score or -1
        is_best = self.__best_score is None or score < self.__best_score - self.eps
        self.__best_score = self.__best_score is not None and min(score, self.__best_score) or score
        if is_best:
            logger.info('Current model is best by val score %.5f < %.5f' % (self.__best_score, prev_best_score))
            self.__early_stop_counter = 0
        else:
            self.__early_stop_counter += 1
            if self.__early_stop_counter >= self.early_stop_n:
                logger.info('Early stopping, regress for %d iterations', self.__early_stop_counter)
                to_break = True
        logger.info('early_stop_counter: %d', self.__early_stop_counter)

        if (self.warm_up_epochs and self.__descrease_times == 0 and self.__warm_up and epoch >= self.warm_up_epochs - 1 ) or \
                (self.__lr_changed <= epoch - self.patience and \
                (self.__early_stop_counter is not None and self.patience and self.__early_stop_counter >= self.patience)):
            self.__lr_changed = epoch
            for param_group in self.optimizer.param_groups:
                if self.__descrease_times == 0 and self.__warm_up:
                    param_group['lr'] = param_group['after_warmup_lr']
                else:
                    param_group['lr'] = param_group['lr'] * self.decrease_rate
                logger.info('Setting for group learning rate=%.8f, epoch=%d', param_group['lr'], self.__lr_changed)
            adjusted = True
            self.__descrease_times += 1

        return adjusted, to_break, is_best


def save_checkpoint(state, epoch, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)
        shutil.copyfile(filename, best_filename + '-%d' % epoch)


def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    predictions = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)
        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i and i % 5 == 0) or i == len(train_loader) - 1:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc=predictions))

    return losses.avg

def validate(val_loader, model, criterion, validation_func=dice_loss, activation=None):
    logger.info('Validating model')
    batch_time = AverageMeter()
    losses = AverageMeter()
    val_funcs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # compute val functions
        vf = validation_func(output, target_var).mean()
        val_funcs.update(vf.data[0], input.size(0))

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Test: [{0}/{0}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.avg:.5f}\t'
          'Val loss: {val_funcs.avg}\t'.format(
           len(val_loader), batch_time=batch_time, loss=losses, val_funcs=val_funcs))

    return losses.avg


def get_outputs(loader, model, activation):
    model.eval()
    outputs, targets = [], []
    for i, (input, target) in enumerate(loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        output = model(input_var)
        if activation is not None:
            output = activation(output)
        outputs.extend(output.cpu().data)
        targets.extend(target)
    return outputs, targets


def test_model(test_loader, model, activation=None):
    logger.info('Testing')
    model.eval()

    names, results = [], []
    for i, (input, name_batch) in enumerate(test_loader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()

        output = model(input_var)
        if activation is not None:
            output = activation(output)

        names.extend(name_batch)
        results.extend(output.data.cpu())
        if i and i % 20 == 0:
            logger.info('Batch %d',i)

    return names, results
