#-*- coding: utf8 -*-
import shutil, time, logging
import torch
import torch.optim
import numpy as np
import visdom, copy
from datetime import datetime
from collections import defaultdict


logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)


def init_optimizer(model, config, exact_layers=None):
    """Sets lr for each layer for specified exact_layers.
    param 'exact_layers' specifies which parameters of the model to train, None - all,
    else - list of layers with a multiplier (optional) for LR schedule"""
    opt_type = config.optimizer
    if exact_layers:
        logger.info('Learning exact layers, number=%d', len(exact_layers))
        parameters = []
        for i, layer in enumerate(exact_layers):
            if isinstance(layer, tuple) and len(layer) == 2:
                layer, multiplier = layer
                init_multiplier = 1
            elif isinstance(layer, tuple) and len(layer) == 3:
                layer, init_multiplier, multiplier = layer
            else:
                multiplier = 1
                init_multiplier = 1
            lr = config.lr * multiplier
            init_lr = config.lr * multiplier * init_multiplier
            logger.info('Layer=%d, lr=%.5f', i, init_lr)
            # warmup is used in PlateauScheduler
            parameters.append({'params': layer.parameters(), 'lr': init_lr, 'after_warmup_lr': lr})
    else:
        logger.info('Optimizing all parameters, lr=%.5f', config.lr)
        parameters = model.parameters()

    if opt_type == 'sgd':
        optimizer = torch.optim.SGD(parameters, config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif opt_type == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise TypeError, 'Unknown optimizer type=%s' % (opt_type, )
    return optimizer


def adjust_lr_schedule(optimizer, epoch, decrease_rate=0.1, each_epochs=10):
    """Sets the learning rate to the initial LR decayed by 1/decrease_rate every <each_epochs> epochs"""
    if not isinstance(optimizer, torch.optim.SGD):
        return
    if epoch and epoch % each_epochs == 0:
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] *= decrease_rate
            logger.info('Setting learning layer=i, rate=%.6f', i, param_group['lr'])


class PlateauScheduler(object):
    """Sets the lr to the initial LR decayed by 1/decrease_rate, when not improving for max_stops epochs

    Warning: after merging schedulers (v0.2) in the stable branch of pytorch, consider using native plateau scheduler
    http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    """
    def __init__(self, optimizer, patience, early_stop_n, decrease_rate=0.1, eps=1e-5,
                 warm_up_epochs=None, best_score=None):
        self.optimizer = optimizer
        if not isinstance(optimizer, (torch.optim.SGD, torch.optim.Adam)):
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
        self.__warm_up = self.__has_warm_up(optimizer)

    def __has_warm_up(self, optimizer):
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] != param_group['after_warmup_lr']:
                logger.info('Optimizer has warm-up stage')
                return True

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