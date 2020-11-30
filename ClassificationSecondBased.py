import importlib
import logging
import numpy as np
import os
import tensorflow as tf
import time
from tqdm import tqdm

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataset import Dataset as Dataset
from ftpdataset import Dataset as FTPDataset

def main(args):
    print("Loading Testing Data:", args.testing)
    print("args.PCA:", args.PCA)
    print("args.featuresVideo:", args.featuresVideo)
    print("args.featuresAudio:", args.featuresAudio)
    print("args.network:",args.network)
    print("args.VLAD_k:",args.VLAD_k)
    print("Architecture:", args.architecture)
    print("args.system:", args.system)
    print("flush!", flush=True)

    if args.FTP:
        dataset = FTPDataset()
    else:
        dataset = Dataset()

    dataset.loadSpottingTestingDataset(
        path_data=args.testing,
        featureVideoName=args.featuresVideo,
        featureAudioName=args.featuresAudio,
        PCA=args.PCA,
        window_size_sec=args.WindowSize
    )

    module = importlib.import_module('networks')
    class_ = getattr(module, args.architecture)
    network = class_(dataset, args.network, VLAD_K=args.VLAD_k)


    create_directory("data")

    with tf.Session() as sess:
        if not os.path.exists(args.tflog):
            os.makedirs(args.tflog)
        network.initialize(sess)

        sess.run(tf.local_variables_initializer())
        sess.run([network.reset_metrics_op])

        start_time = time.time()
        total_num_batches = 0
        for i in tqdm(range(dataset.nb_batch_testing)):
            batch_video_feature, batch_audio_features, batch_labels, key = dataset.getSpottingTestingBatch(i)


            feed_dict = {
                network.video_input: batch_video_feature[:batch_video_feature.shape[0]//4,:,:],
                network.audio_input: batch_audio_features[:batch_audio_features.shape[0]//4,:,:],
                network.labels: batch_labels[:batch_labels.shape[0]//4,:],
                network.keep_prob: 1.0,
                network.weights: dataset.weights
            }

            sess.run([network.loss], feed_dict=feed_dict)
            sess.run([network.update_metrics_op], feed_dict=feed_dict)
            vals_test = sess.run(network.metrics_op, feed_dict=feed_dict)
            predictions_1 = sess.run(network.predictions, feed_dict=feed_dict)

            feed_dict = {
                network.video_input: batch_video_feature[batch_video_feature.shape[0]//4:2*batch_video_feature.shape[0]//4,:,:],
                network.audio_input: batch_audio_features[batch_audio_features.shape[0]//4:2*batch_audio_features.shape[0]//4,:,:],
                network.labels: batch_labels[batch_labels.shape[0]//4:2*batch_labels.shape[0]//4,:],
                network.keep_prob: 1.0,
                network.weights: dataset.weights
            }

            sess.run([network.loss], feed_dict=feed_dict)
            sess.run([network.update_metrics_op], feed_dict=feed_dict)
            vals_test = sess.run(network.metrics_op, feed_dict=feed_dict)
            predictions_2 = sess.run(network.predictions, feed_dict=feed_dict)

            feed_dict = {
                network.video_input: batch_video_feature[2*batch_video_feature.shape[0]//4:3*batch_video_feature.shape[0]//4,:,:],
                network.audio_input: batch_audio_features[2*batch_audio_features.shape[0]//4:3*batch_audio_features.shape[0]//4,:,:],
                network.labels: batch_labels[2*batch_labels.shape[0]//4:3*batch_labels.shape[0]//4,:],
                network.keep_prob: 1.0,
                network.weights: dataset.weights
            }

            sess.run([network.loss], feed_dict=feed_dict)
            sess.run([network.update_metrics_op], feed_dict=feed_dict)
            vals_test = sess.run(network.metrics_op, feed_dict=feed_dict)
            predictions_3 = sess.run(network.predictions, feed_dict=feed_dict)

            feed_dict = {
                network.video_input: batch_video_feature[3*batch_video_feature.shape[0]//4:,:,:],
                network.audio_input: batch_audio_features[3*batch_audio_features.shape[0]//4:,:,:],
                network.labels: batch_labels[3*batch_labels.shape[0]//4:,:],
                network.keep_prob: 1.0,
                network.weights: dataset.weights
            }

            sess.run([network.loss], feed_dict=feed_dict)
            sess.run([network.update_metrics_op], feed_dict=feed_dict)
            vals_test = sess.run(network.metrics_op, feed_dict=feed_dict)
            predictions_4 = sess.run(network.predictions, feed_dict=feed_dict)



            predictions = np.concatenate((predictions_1, predictions_2, predictions_3, predictions_4))
            mean_error = np.mean(np.abs(predictions-batch_labels), axis=0)
            if args.system == "windows":
                key = key.replace("/", "\\")
                sep = "\\"
            else:
                sep = "/"

            if args.FTP:
                path = key
            else:
                ind = key.index("data") + len("data") + 1
                path = os.path.split(key)[0][len('feats/SoccerNet/data/'):]
                path = key[ind:]

            directory_name = "data"
            for f in path.split(sep):
                directory_name = os.path.join(directory_name, f)
                create_directory(directory_name)
            predictions_name = os.path.join(directory_name, args.output + "_" + os.path.split(key)[1])
            np.save(predictions_name, predictions)

            total_num_batches += 1

            vals_test['mAP'] = np.mean([vals_test['auc_PR_1'], vals_test['auc_PR_2'], vals_test['auc_PR_3']])

        good_sample = np.sum( np.multiply(vals_test['confusion_matrix'], np.identity(4)), axis=0)
        bad_sample = np.sum( vals_test['confusion_matrix'] - np.multiply(vals_test['confusion_matrix'], np.identity(4)), axis=0)
        vals_test['accuracies'] =  good_sample / ( bad_sample + good_sample )
        vals_test['accuracy'] = np.mean(vals_test['accuracies'])

        print(vals_test['confusion_matrix'])
        print(('auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)') %
        (vals_test['auc_PR'], vals_test['auc_PR_0'], vals_test['auc_PR_1'], vals_test['auc_PR_2'], vals_test['auc_PR_3']))
        print(' Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}'.format(vals_test['loss'], vals_test['accuracy'], vals_test['mAP']))
        print(' Time: {:<8.3} s'.format(time.time()-start_time))

    if args.FTP:
        dataset.remove_files()

def create_directory(dir_name):
    """Creates a directory if it does not exist
    Parameters
    ----------
    dir_name : str
        The name of the directory to create
    """
    if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
            os.mkdir(dir_name)


if __name__ == "__main__":
    parser = ArgumentParser(description='', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--testing',    required=False,  type=str,   default='/media/giancos/Football/dataset_crop224/listgame_Test_2.npy',  help='the file containg the testing data.')
    parser.add_argument('--featuresVideo',   required=False, type=str,   default="ResNET",  help='select typeof features video')
    parser.add_argument('--featuresAudio',   required=False, type=str,   default="VGGish",  help='select typeof features audio')
    parser.add_argument('--architecture', required=True, type=str, help='the name of the architecture to use.')
    parser.add_argument('--network',    required=False, type=str,   default="RVLAD",     help='Select the type of network (CNN, MAX, AVERAGE, VLAD)')
    parser.add_argument('--VLAD_k',     required=False, type=int,   default=64,     help='number of cluster for slustering method (NetVLAD, NetRVLAD, NetDBOW, NetFV)' )
    parser.add_argument('--WindowSize', required=False, type=int,   default=60,     help='Size of the Window' )
    parser.add_argument('--output', required=False, type=str, default="", help="Prefix for the output files.")
    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--PCA',        required=False, action="store_true",        help='use PCA version of the features')
    parser.add_argument('--tflog',      required=False, type=str,   default='Model',   help='folder for tensorBoard output')
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')
    parser.add_argument('--system', required=False, type=str, default="linux", help='linux or windows')
    parser.add_argument('--FTP', required=False, action="store_true",        help='use FTP version of the dataset')


    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    args.PCA = True

    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=numeric_level)
    delattr(args, 'loglevel')
    if (args.GPU >= 0):
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    start=time.time()
    main(args)
    logging.info('Total Execution Time is {0} seconds'.format(time.time()-start))
