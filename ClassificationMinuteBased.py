import importlib
import logging
import time

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ftpdataset import Dataset
from networks import AudioNetwork, VideoNetwork
from trainer import Trainer

def main(args):
    print("Architecture:", args.architecture)
    print("Loading Training Data:", args.training)
    print("args.featuresVideo:", args.featuresVideo)
    print("args.featuresAudio:", args.featuresAudio)
    print("args.PCA:", args.PCA)
    print("args.network:",args.network)
    print("args.LR:",args.LR)
    print("args.VLAD_k:",args.VLAD_k)
    print("args.max_epoch:",args.max_epoch)
    print("args.WindowSize:", args.WindowSize)
    print("args.featurePerSec:", args.featurePerSec)
    print("args.mode:", args.mode)
    print("args.config:", args.config)
    print("args.system:", args.system)
    print("flush!", flush=True)

    dataset = Dataset(args.batch_size, system=args.system)

    if args.mode == "train":
        if args.formatdataset == 1:
            dataset.prepareTrainingDataset(
                path_data = args.training,
                featureVideoName = args.featuresVideo,
                featureAudioName = args.featuresAudio,
                PCA = args.PCA,
                window_size_sec = args.WindowSize,
                feature_per_second = args.featurePerSec
            )
            dataset.prepareValidationDataset(
                path_data = args.validation,
                featureVideoName = args.featuresVideo,
                featureAudioName = args.featuresAudio,
                PCA = args.PCA,
                window_size_sec = args.WindowSize,
                feature_per_second = args.featurePerSec
            )
        else:
            dataset.loadTrainingDataset(
                path_data = args.training,
                featureVideoName = args.featuresVideo,
                featureAudioName = args.featuresAudio,
                PCA = args.PCA,
                window_size_sec = args.WindowSize,
                feature_per_second = args.featurePerSec
            )

            dataset.loadValidationDataset(
                path_data = args.validation,
                featureVideoName = args.featuresVideo,
                featureAudioName = args.featuresAudio,
                PCA = args.PCA,
                window_size_sec = args.WindowSize,
                feature_per_second = args.featurePerSec
            )

    if args.formatdataset == 1:
        dataset.prepareTestingDataset(
            path_data = args.testing,
            featureVideoName = args.featuresVideo,
            featureAudioName = args.featuresAudio,
            PCA = args.PCA,
            window_size_sec = args.WindowSize,
            feature_per_second = args.featurePerSec
        )
    else:
        dataset.loadTestingDataset(
            path_data = args.testing,
            featureVideoName = args.featuresVideo,
            featureAudioName = args.featuresAudio,
            PCA = args.PCA,
            window_size_sec = args.WindowSize,
            feature_per_second = args.featurePerSec
        )

    module = importlib.import_module('networks')
    class_ = getattr(module, args.architecture)
    network = class_(dataset, args.network, VLAD_K=args.VLAD_k)

    trainer = Trainer(dataset, network, output_prefix=args.outputPrefix)
    if args.mode == "predict":
        f = open(args.config)
        lines = f.readlines()
        properties = [line.replace('\n', '') for line in lines]
        for prop in properties:
            trainer.predict(prop, display=prop==properties[-1], tflog=args.tflog)
    elif args.mode == "train":
        trainer.train(epochs=args.max_epoch, learning_rate=args.LR, tflog=args.tflog)
    else:
        trainer.predict_other()

if __name__ == '__main__':
    parser = ArgumentParser(description='', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--architecture', required=True, type=str, help='the name of the architecture to use.')
    parser.add_argument('--training',   required=True,  type=str,   help='the file containg the training data.')
    parser.add_argument('--validation', required=True,  type=str,   help='the file containg the validation data.')
    parser.add_argument('--testing',    required=True,  type=str,   help='the file containg the validation data.')
    parser.add_argument('--featuresVideo',   required=False, type=str,   default="ResNET",  help='select typeof features video')
    parser.add_argument('--featuresAudio',   required=False, type=str,   default="VGGish",  help='select typeof features audio')
    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--PCA',        required=False, action="store_true",        help='use PCA version of the features')
    parser.add_argument('--network',    required=False, type=str,   default="RVLAD",     help='Select the type of network (CNN, MAX, AVERAGE, VLAD)')
    parser.add_argument('--tflog',      required=False, type=str,   default='Model',   help='folder for tensorBoard output')
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')
    parser.add_argument('--batch_size', required=False, type=int,   default=60,     help='Size of the batch in number of halves game' )
    parser.add_argument('--max_epoch',  required=False, type=int,   default=200,    help='maximum number of epochs' )
    parser.add_argument('--VLAD_k',     required=False, type=int,   default=64,     help='number of cluster for slustering method (NetVLAD, NetRVLAD, NetDBOW, NetFV)' )
    parser.add_argument('--LR',         required=False, type=float, default=0.01,   help='Learning Rate' )
    parser.add_argument('--WindowSize', required=False, type=int,   default=60,     help='Size of the Window' )
    parser.add_argument('--featurePerSec', required=False, type=int, default=2, help='Number of features per second')
    parser.add_argument('--outputPrefix', required=False, type=str, default="", help="Prefix for the output files.")
    parser.add_argument('--mode', required=False, type=str, default="train", help="train or predict or other. With predict, you can use the --config option.")
    parser.add_argument('--config', required=False, type=str, default="None", help="config file with property for which the output must be saved. Must be used with --mode predict option.")
    parser.add_argument('--formatdataset', required=False, type=int, default=0, help='1 if you need to prepare the dataset, 0 else.')
    parser.add_argument('--system', required=False, type=str, default="linux", help='linux or windows')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)


    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=numeric_level)
    delattr(args, 'loglevel')
    if (args.GPU >= 0):
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    start = time.time()
    main(args)
    logging.info('Total Execution Time is {0} seconds'.format(time.time()-start))
