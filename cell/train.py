#-*- coding: utf8 -*-
import sys, random, os, logging, glob, traceback, cv2
from collections import defaultdict
from optparse import OptionParser
import numpy as np, pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler
import torch, torch.nn as nn, torch.nn.parallel, torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from folder import ImageFolder, ImageTestFolder
from __init__ import RESULT_DIR
import model, transform_rules
from utils import VisdomMonitor, dice_loss, rle_encoding, prob_to_rles
from model import factory as model_factory
from transform_rules import augmentation_factory
import routine


logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)


class Trainer(object):
    def __init__(self, config, train_dir, test_dir, seed=0,
                 shard='', pre_trained=True):
        self.__transformations = augmentation_factory(config.transform)
        self.__train_dir = train_dir
        self.__test_dir = test_dir
        self.__seed = seed
        self.__pre_trained = pre_trained
        res_dir = RESULT_DIR.format(shard)
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)

        # modify logs for sharding (parallel models on different GPUs)
        if shard != '':
            fh = logging.FileHandler(os.path.join(res_dir, 'application.log'))
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
            fh.setFormatter(formatter)
            logger.handlers[0] = fh

        self.__checkpoint_file = os.path.join(res_dir, 'checkpoint{0}.pth.tar')
        self.__best_model_filename = os.path.join(res_dir, 'model_best{0}.pth.tar')
        self.__output_file = os.path.join(res_dir, 'submission{0}.csv')
        self.__detailed_output_file = os.path.join(res_dir, 'detailed_submission{0}.txt')
        self.__holdout_output_file = os.path.join(res_dir, 'detailed_holdout{0}.txt')
        self.__valid_output_file = os.path.join(res_dir, 'detailed_valid{0}.txt')
        self._init_model(config)
        self.monitor = VisdomMonitor(port=80)
        self.__cur_fold = ''

    def __rm_prev_checkpoints(self):
        for fname in glob.glob(self.__best_model_filename.format(self.__cur_fold) + '-*'):
            os.remove(fname)

    def _init_model(self, config):
        logger.info('Initing model')
        model = model_factory(config.model)
        if not torch.has_cudnn:
            raise RuntimeError('The model in CPU mode, the code is designed for cuda only')

        #if not isinstance(model, torch.nn.DataParallel):
        #    model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
        self._model = model
        self.__layers_to_optimize = None

    def _init_checkpoint(self, optimizer, config):
        start_epoch = 0
        best_score = None

        if config.from_checkpoint:
            checkpoint = routine.load_checkpoint(self.__checkpoint_file.format(self.__cur_fold))
            start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            self._model.load_state_dict(checkpoint['state_dict'])
            # Sometimes cause error b/c of multiple param groups, pytorch bug ?
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('Resuming from checkpoint epoch=%d, score=%.5f' % (start_epoch, best_score or -1))

        return start_epoch, best_score

    def _save_checkpoint(self, config, optimizer, epoch, is_best, best_score):

        state = {
            'epoch': epoch + 1,
            'state_dict': self._model.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
            'arch': config.model
        }

        routine.save_checkpoint(state, epoch, is_best, filename=self.__checkpoint_file.format(self.__cur_fold),
                                    best_filename=self.__best_model_filename.format(self.__cur_fold))

    def _split_data(self, folder, train_percent):
        random.seed(self.__seed)
        train_names, val_names = set(), set()

        for filename in os.listdir(folder):
            ddict = train_names if random.random() < train_percent else val_names
            ddict.add(filename)

        logger.info('Splitted dataset: train %d, val %d', len(train_names), len(val_names))
        return train_names, val_names

    def _get_data_loader(self, config, names=None):
        trs = self.__transformations
        if names is None:
            train_names, val_names = self._split_data(self.__train_dir, config.train_percent)
        else:
            train_names, val_names = names

        train_folder = ImageFolder(self.__train_dir, train_names, transform=trs['train'])
        val_folder = ImageFolder(self.__train_dir, val_names, transform=trs['val'])
        if not len(train_folder) or not len(val_folder):
            raise ValueError('One of the image folders contains zero data, train: %s, val: %s' % \
                              (len(train_folder), len(val_folder)))

        sampler = None
        train_loader = torch.utils.data.DataLoader(train_folder, batch_size=config.batch_size, shuffle=True,
                                                   sampler=sampler, num_workers=config.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_folder, batch_size=config.batch_size, shuffle=True,
                                                 num_workers=config.workers, pin_memory=True)

        return train_loader, val_loader

    def _init_loss(self):
        criterion = dice_loss # TODO: to cuda ?
        return criterion

    def _init_lr_scheduler(self, config, optimizer, best_score):
        lr_scheduler = routine.PlateauScheduler(optimizer, config.max_stops, config.early_stop_n,
                                                    decrease_rate=config.decrease_rate,
                                                    best_score=best_score)
        if config.lr_schedule == 'frozen':
            lr_scheduler.patience = 99999

        return lr_scheduler

    def _load_best_model(self):
        fname = self.__best_model_filename.format(self.__cur_fold)
        checkpoint = routine.load_checkpoint(fname)
        self._model.load_state_dict(checkpoint['state_dict'])
        logger.info('Loaded best model %s, arch=%s', fname, checkpoint.get('arch', ''))

    def train_single_model(self, config, train_loader, val_loader):
        # variables: split, maybe some params
        logger.info('Starting learning single model')
        #self._init_model() ???
        self.__rm_prev_checkpoints()

        optimizer = torch.optim.SGD(self._model.parameters(), config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
        start_epoch, best_score = self._init_checkpoint(optimizer, config)
        lr_scheduler = self._init_lr_scheduler(config, optimizer, best_score)

        criterion = self._init_loss()

        for epoch in range(start_epoch, config.epoch):
            logger.info('Epoch %d', epoch)
            # iterate
            train_score = routine.train(train_loader, self._model, criterion, optimizer, epoch)
            score = routine.validate(val_loader, self._model, criterion, activation=None)
            self.monitor.add_performance('loss', train_score, score)

            # change lr and save checkpoint
            adjusted, to_break, is_best = lr_scheduler.step(epoch, score)
            self._save_checkpoint(config, optimizer, epoch, is_best, best_score)
            if to_break:
                logger.info('Exiting learning process')
                break
            # load model if plateau == True
            if adjusted and config.lr_schedule == 'adaptive_best':
                logger.info('Loaded the model from previous best step')
                self._load_best_model()

    def run(self, config):
        train_loader, val_loader = self._get_data_loader(config)
        self.train_single_model(config, train_loader, val_loader)
        self.test_and_submit(config, val_loader)

    def run_ensemble(self, config):
        raise NotImplementedError

    def test_model(self, config):
        tr = self.__transformations['test']
        test_folder = ImageTestFolder(self.__test_dir, transform=tr)

        test_loader = torch.utils.data.DataLoader(test_folder, batch_size=config.test_batch_size,
                                                      num_workers=config.workers, pin_memory=True)

        names, results = routine.test_model(test_loader, self._model, activation=None)



        # sys.exit(path)
        # print ('TODO')
        # # TODO: rewrite
        # crop_num = len(tr.transforms[0])
        # for index in range(crop_num):
        #     # iterate over tranformations
        #     logger.info('Testing transformation %d/%d', index + 1, crop_num)
        #     test_folder.transform.transforms[0].index = index
        #     test_loader = torch.utils.data.DataLoader(test_folder, batch_size=config.test_batch_size,
        #                                               num_workers=config.workers, pin_memory=True)
        #     names, crop_results = routine.test_model(test_loader, self._model, activation=None)
        #     results.append(crop_results)
        #
        # final_results = [sum(map(lambda x: x[i].data.numpy(), results)) / float(crop_num) for i in
        #                  range(len(test_folder.imgs))]


        final_results = [r.squeeze().numpy() for r in results]
        return names, final_results

    def __write_submission(self, names, results, threshold = 0.5, output_file=None):
        # TODO: predict thresolhd
        # TODO: +postprocessing
        output_file = output_file or self.__output_file.format(self.__cur_fold)

        with open(output_file, 'w') as wf:
            wf.write('ImageId,EncodedPixels\n')
            for basename, vector in zip(names, results):
                filepath = os.path.join(self.__test_dir, basename)
                name = basename.split('/')[0]
                target_shape = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE).shape
                img = cv2.resize(vector, (target_shape[1], target_shape[0]))
                #img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)[1]
                #print(name, vector.shape, target_shape, img.shape)
                #cv2.imwrite('/home/tyantov/t3.png', img * 255)
                #img = cv2.imread(self.__train_dir + '/00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552/mask/one_mask.png', cv2.IMREAD_GRAYSCALE)/255

                for r_mask in prob_to_rles(img, cut_off=threshold):
                    str_t = name + ',' + ' '.join(map(str, r_mask)) + '\n'
                    wf.write(str_t)

    def test_and_submit(self, config, val_loader=None):
        #self._init_model()
        self._load_best_model()
        names, final_results = self.test_model(config)
        self.__write_submission(names, final_results)


def main():
    parser = OptionParser()
    parser.add_option("-b", "--batch_size", action='store', type='int', dest='batch_size', default=64)
    parser.add_option("--test_batch_size", action='store', type='int', dest='test_batch_size', default=320)
    parser.add_option("-e", "--epoch", action='store', type='int', dest='epoch', default=80)
    parser.add_option("-r", "--workers", action='store', type='int', dest='workers', default=2)
    parser.add_option("-l", "--learning-rate", action='store', type='float', dest='lr', default=0.01)
    parser.add_option("-m", "--momentum", action='store', type='float', dest='momentum', default=0.9)
    parser.add_option("-w", "--weight_decay", action='store', type='float', dest='weight_decay', default=1e-4)
    parser.add_option("-o", "--optimizer", action='store', type='string', dest='optimizer', default='sgd',
                      help='sgd|adam|yf')
    parser.add_option("-c", "--from_checkpoint", action='store', type='int', dest='from_checkpoint', default=0,
                      help='resums training from a specific epoch')
    parser.add_option("--train_percent", action='store', type='float', dest='train_percent', default=0.8,
                      help='train/val split percantage')
    parser.add_option("--decrease_rate", action='store', type='float', dest='decrease_rate', default=0.1,
                      help='For lr schedule, on plateau how much to descrease lr')
    parser.add_option("--early_stop_n", action='store', type='int', dest='early_stop_n', default=6,
                      help='Early stopping on a specific number of degrading epochs')
    parser.add_option("--folds", action='store', type='int', dest='folds', default=10,
                      help='Number of folds, for ensemble training only')
    parser.add_option("--lr_schedule", action='store', type='str', dest='lr_schedule', default='adaptive_best',
                      help="""possible: adaptive, adaptive_best, decreasing or frozen.
                       adaptive_best is the same as plateau scheduler""")
    parser.add_option("--model", action='store', type='str', dest='model', default='unet',
                      help='Which model to use, check model.py for names')
    parser.add_option("--transform", action='store', type='str', dest='transform', default='np_nozoom_256',
                      help='Specify a transformation rule. Check transform_rules.py for names')
    parser.add_option("--warm_up_epochs", action='store', type='int', dest='warm_up_epochs', default=2,
                      help='warm_up_epochs number if model has it')
    parser.add_option("--max_stops", action='store', type='int', dest='max_stops', default=2,
                      help='max_stops for plateau/adaptive lr schedule')
    parser.add_option("--run_type", action='store', type='str', dest='run_type', default='eval',
                      help='train|eval|ens')
    parser.add_option("--seed", action='store', type='int', dest='seed', default=0)
    parser.add_option("--shard", action='store', type='str', dest='shard', default='',
                      help='Postfix for results folder, where the results will be saved, <results+shard>/')

    # Options
    config, _ = parser.parse_args()
    assert config.lr_schedule in ('adaptive', 'decreasing', 'frozen', 'adaptive_best')
    # Init trainer
    from __init__ import TRAIN_FOLDER, TEST_FOLDER
    train_folder = TRAIN_FOLDER
    test_folder = TEST_FOLDER

    tr = Trainer(config, train_folder, test_folder, seed=config.seed, shard=config.shard)
    logger.info('Config: %s', str(config))
    # Run
    try:
        if config.run_type == 'train':
            tr.run(config)
        elif config.run_type == 'eval':
            tr.test_and_submit(config)
        elif config.run_type == 'ens':
            tr.run_ensemble(config)
    except:
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    sys.exit(main())
