from __future__ import division

import os, sys, time, random, copy
import torch
import argparse
import torch.backends.cudnn as cudnn

from utils.data import get_data_specs, get_data
from utils.utils import get_model_path, time_string, convert_secs2time, print_log
from utils.network import get_network, get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters
from utils.training import AverageMeter, RecorderMeter
from utils.training import adjust_learning_rate, train_target_model, validate, save_checkpoint, accuracy

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a neural network')

    # pretrained
    parser.add_argument('--pretrained_dataset', default='cifar10', choices=['mnist', 'cifar10', 'cifar100', 'gtsrb', 'eurosat', 'ycb'],
                        help='Used pretrained_dataset (default: cifar10)')
    parser.add_argument('--pretrained_arch', default='resnet18', choices=['conv_net', 'lenet', 'resnet18', 'vgg16_cifar', 'vgg19_cifar', 'resnet20', 'resnet50', 'resnet56', 'vgg16', 'vgg19', 'resnet152', 'mobilenet_v2', "inception_v3"],
                        help='Used model architecture for the grad imgs generation process: (default: resnet18)')
    parser.add_argument('--pretrained_seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')
    parser.add_argument('--finetune', action='store_true',
                        help='Finetune from pretrained imagenet weight (default: False)')
    # Optimization options
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (dfault: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning Rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum (default: 0.9)')
    parser.add_argument('--decay', type=float, default=0.0005,
                        help='Weight decay (L2 penalty) (default: 0.0005)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[],
                        help='Decrease learning rate at these epochs (default: [])')
    parser.add_argument('--gammas', type=float, nargs='+', default=[],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule (default: [])')
    parser.add_argument('--print_freq', default=200, type=int, metavar='N',
                        help='print frequency (default: 200)')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of data loading workers (default: 6)')
    args = parser.parse_args()

    args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

    if args.pretrained_seed is None:
        args.pretrained_seed = random.randint(1, 10000)

    return args

def main():
    args = parse_arguments()

    random.seed(args.pretrained_seed)
    torch.manual_seed(args.pretrained_seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.pretrained_seed)
    cudnn.benchmark = True

    # get a path for saving the model to be trained
    model_path = get_model_path(dataset_name=args.pretrained_dataset,
                                network_arch=args.pretrained_arch,
                                random_seed=args.pretrained_seed)

    # Init logger
    log_file_name = os.path.join(model_path, 'log_seed_{}.txt'.format(args.pretrained_seed))
    print("Log file: {}".format(log_file_name))
    log = open(log_file_name, 'w')
    print_log('save path : {}'.format(model_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        print_log("{} : {}".format(key, value), log)
    print_log("Random Seed: {}".format(args.pretrained_seed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("Torch  version : {}".format(torch.__version__), log)
    print_log("Cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    # Get data specs
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.pretrained_dataset, args.pretrained_arch)
    pretrained_data_train, pretrained_data_test = get_data(args.pretrained_dataset,
                                                            mean=mean,
                                                            std=std,
                                                            input_size=input_size,
                                                            train_target_model=True)

    pretrained_data_train_loader = torch.utils.data.DataLoader(pretrained_data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    pretrained_data_test_loader = torch.utils.data.DataLoader(pretrained_data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)


    print_log("=> Creating model '{}'".format(args.pretrained_arch), log)
    # Init model, criterion, and optimizer
    net = get_network(args.pretrained_arch, input_size=input_size, num_classes=num_classes, finetune=args.finetune)
    print_log("=> Network :\n {}".format(net), log)
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    non_trainale_params = get_num_non_trainable_parameters(net)
    trainale_params = get_num_trainable_parameters(net)
    total_params = get_num_parameters(net)
    print_log("Trainable parameters: {}".format(trainale_params), log)
    print_log("Non Trainable parameters: {}".format(non_trainale_params), log)
    print_log("Total # parameters: {}".format(total_params), log)

    # define loss function (criterion) and optimizer
    criterion_xent = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)

    if args.use_cuda:
        net.cuda()
        criterion_xent.cuda()

    recorder = RecorderMeter(args.epochs)

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.epochs):
        current_learning_rate = adjust_learning_rate(args.learning_rate, args.momentum, optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train_target_model(pretrained_data_train_loader, net, criterion_xent, optimizer, epoch, log,
                                    print_freq=args.print_freq,
                                    use_cuda=args.use_cuda)

        # evaluate on validation set
        print_log("Validation on pretrained test dataset:", log)
        val_acc = validate(pretrained_data_test_loader, net, criterion_xent, log, use_cuda=args.use_cuda)
        is_best = recorder.update(epoch, train_los, train_acc, 0., val_acc)

        save_checkpoint({
          'epoch'       : epoch + 1,
          'arch'        : args.pretrained_arch,
          'state_dict'  : net.state_dict(),
          'recorder'    : recorder,
          'optimizer'   : optimizer.state_dict(),
          'args'        : copy.deepcopy(args),
        }, model_path, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(model_path, 'curve.png') )

    log.close()

if __name__ == '__main__':
    main()
