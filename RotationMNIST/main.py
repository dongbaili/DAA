# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import os
import sys
from copy import deepcopy
from collections import defaultdict

from math import ceil
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed.datasets import get_dataloaders, num_classes_dict, checkpoint_step_dict, train_steps_dict
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.Logger import Logger
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_args():
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--txtdir', type=str, default="domainbed/txtlist")
    parser.add_argument('--dataset', type=str, default="RMnist")
    parser.add_argument("--source", nargs='+')
    parser.add_argument("--target", nargs="+")
    parser.add_argument("--my_algorithm", type=str, choices=["ERM", "Reweight", "SelfSupervise", "WDAN"])

    # Do not change the following parameters, they are irrelevant of this experiment (just stay default all the time)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--mix', action='store_true')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--pretrain', default="Supervised")
    parser.add_argument('--linear_probe', action='store_true')
    parser.add_argument('--trainable_layers_start', type=int, default=0)
    parser.add_argument('--arch', default="resnet50")
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--scheduler', type=str, default="None")
    parser.add_argument('--steps', type=int, default=0,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--hparams_str', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_fixed_config', type=str,
        help='config of hparams (fixed value, not random)')
    parser.add_argument('--hparams_rand_config', type=str,
        help='Path of json config for random hparams generation')
    parser.add_argument('--data_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--checkpoint_step_freq', type=int, default=0,
        help='Checkpoint every N steps.')
    parser.add_argument('--stepval_freq', type=int, default=20,
        help='print step val every N steps.')
    parser.add_argument('--log_dir', type=str, default="logs")
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--load_model_best', action='store_true')
    parser.add_argument('--result_name', type=str)

    args = parser.parse_args()
    misc.setup_seed(args.seed)

    if args.checkpoint_step_freq == 0:
        args.checkpoint_step_freq = checkpoint_step_dict[args.dataset]
    if args.steps == 0:
        args.steps = train_steps_dict[args.dataset]
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    if args.result_name:
        os.makedirs(os.path.join(args.output_dir, args.result_name), exist_ok=True)
        cur_seed = misc.seed_hash(args.seed, args.data_seed, args.hparams_seed)
    if args.hparams_str:
        hparams.update(json.loads(args.hparams_str))
    elif args.hparams_fixed_config:
        with open(args.hparams_fixed_config) as f:
            hparams.update(json.load(f)) 
    elif args.hparams_rand_config:
        hparams.update(hparams_registry.load_config(args.hparams_rand_config, cur_seed))
        if args.result_name:
            hparams_file = os.path.join(args.output_dir, args.result_name, "%d_hparams.json"% cur_seed)
            metrics_file = os.path.join(args.output_dir, args.result_name, "%d_metrics.json"% cur_seed)
            if os.path.exists(metrics_file):
                print("This experiment has alread been runned before")
                sys.exit(0)
            with open(hparams_file, 'w') as f:
                json.dump(hparams, f)

    # hard coding
    hparams["pretrain"] = args.pretrain
    hparams["linear_probe"] = args.linear_probe
    hparams["trainable_layers_start"] = args.trainable_layers_start
    hparams["arch"] = args.arch
    hparams["optimizer"] = args.optimizer
    hparams["scheduler"] = args.scheduler
    hparams["do_ms"] = True if args.algorithm == "MixStyle" else False
    if args.optimizer == "Adahessian":
        hparams["lr"] = hparams["lr"]*100

    return args, hparams


if __name__ == "__main__":
    args, hparams = load_args()
    logger = Logger(args, hparams)
    logger.info("Environment:")
    logger.info("\t`P`ython: {}".format(sys.version.split(" ")[0]))
    logger.info("\tPyTorch: {}".format(torch.__version__))
    logger.info("\tTorchvision: {}".format(torchvision.__version__))
    logger.info("\tCUDA: {}".format(torch.version.cuda))
    logger.info("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.info("\tNumPy: {}".format(np.__version__))
    logger.info("\tPIL: {}".format(PIL.__version__))
    logger.info('Args:')
    for k, v in sorted(vars(args).items()):
        logger.info('\t{}: {}'.format(k, v))

    logger.info('HParams:')
    for k, v in sorted(hparams.items()):
        logger.info('\t{}: {}'.format(k, v))

    if args.load_model_best:
        load_model_path = os.path.join(args.output_dir, Logger.get_expname(args, hparams), 'model_best_seed%d.pkl' % args.seed)
        algorithm_dict = torch.load(load_model_path, map_location="cpu")["model_dict"]
        logger.info("Load model from %s" % load_model_path)
    else:
        algorithm_dict = None
        logger.info("Do not load model")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loaders, val_loaders, test_loaders = get_dataloaders(args, hparams, logger)

    steps_per_epoch = ceil(min([len(train_loader.dataset)/hparams['batch_size'] for train_loader in train_loaders]))
    hparams["epochs"] = ceil(args.steps/steps_per_epoch)
    logger.info("Steps per epoch: %d, number of epochs: %d, number of steps: %d" % (steps_per_epoch, hparams["epochs"], args.steps))

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class((3, 28, 28), num_classes_dict[args.dataset],
        len(args.source), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)

    train_accs = []
    val_accs = dict()
    test_accs = dict()
    best_val_accs = defaultdict(float)
    best_test_accs = dict()

    for step in range(args.steps):
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        
        if args.my_algorithm == "ERM":
            step_vals = algorithm.ERM_update(minibatches_device) # ERM
        elif args.my_algorithm == "Reweight":
            step_vals = algorithm.Reweight_update(minibatches_device, test_loaders, device) # Reweight
        elif args.my_algorithm == "WDAN":
            if step < 2000:
                step_vals = algorithm.ERM_update(minibatches_device)
            else:
                step_vals = algorithm.WDAN_update(minibatches_device, test_loaders[0], device) # WDAN
        elif args.my_algorithm == "SelfSupervise":
            if step < args.steps - 1000:
                step_vals = algorithm.ERM_update(minibatches_device)
            else:
                alpha = step / args.steps
                bar = 0.9
                if step % 20 == 0:
                    pseudo_x, pseudo_y = algorithm.select_pseudo(test_loaders[0], bar, device)
                step_vals = algorithm.SelfSupervise_update(minibatches_device, pseudo_x, pseudo_y, alpha, device)
        else:
            raise NotImplementedError
        
        step_val_str = ""
        for key, val in step_vals.items():
            if key != "train_acc":
                step_val_str = step_val_str + "%s: %.3f, " % (key, val)
        train_accs.append(step_vals["train_acc"])

        epoch = int((step+1) / steps_per_epoch)
        if (step+1) % args.stepval_freq == 0:
            logger.info("Step %d, Epoch %d, %s train acc: %.4f" % (step+1, epoch, step_val_str, np.mean(np.array(train_accs))))
            train_accs = []
            
        if (step+1) % steps_per_epoch == 0:
            algorithm.scheduler_step()
            # logger.info("Next lr: %.8f" % (algorithm.get_lr()[0]))

        if (step+1) % args.checkpoint_step_freq == 0 or step == args.steps-1:
            # validation 
            logger.info("Start validation...")
            correct_overall = 0
            total_overall = 0
            val_loss_overall = 0.0
            for index, domain in enumerate(args.source):
                acc, correct, loss, loss_sum, total = misc.accuracy_and_loss(algorithm, val_loaders[index], None, device)
                correct_overall += correct
                total_overall += total
                val_loss_overall += loss_sum
                val_accs[domain] = acc
            val_accs["overall"] = correct_overall / total_overall
            for k, v in val_accs.items():
                logger.info("Val %s: %.4f" % (k, v))
            val_loss = val_loss_overall / total_overall
            logger.info("Val loss: %.4f" % val_loss)

            # test
            logger.info("Start testing...")
            result_dict = {'intermediate':[], 'final': 0.0}
            correct_overall = 0
            total_overall = 0
            for index, domain in enumerate(args.target):
                acc, correct, total = misc.accuracy(algorithm, test_loaders[index], None, device)
                correct_overall += correct
                total_overall += total
                test_accs[domain] = acc
            test_accs["overall"] = correct_overall / total_overall
            for k, v in test_accs.items():
                logger.info("Test %s: %.4f" % (k, v))
            # nni.report_intermediate_result({"default": val_accs["overall"], "test": test_accs["overall"]})
            result_dict['intermediate'].append({"default": val_accs["overall"], "test": deepcopy(test_accs)})

            if (step+1) / args.checkpoint_step_freq == 1 or test_accs["overall"] > best_test_accs['overall']:
                logger.info("New best validation acc at step %d epoch %d!" % (step+1, epoch))
                best_val_accs = deepcopy(val_accs)
                best_test_accs = deepcopy(test_accs)

            logger.info("")

    logger.info("Final result")
    for k, v in best_val_accs.items():
        logger.info("Best val %s: %.4f" % (k, v))
    for k, v in best_test_accs.items():
        logger.info("Best test %s: %.4f" % (k, v))

    with open("results/accuracies.txt", 'a') as f:
        f.write(f"{args.my_algorithm}")
        f.write(f"{args.source},")
        for k, v in best_val_accs.items():
            f.write("%s: %.4f," % (k, v))
        for k, v in best_test_accs.items():
            f.write("%s: %.4f," % (k, v))
        f.write("\n")