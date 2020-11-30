import loupe as lp
import numpy as np
import random
import tensorflow as tf

from tensorboard.plugins.pr_curve import summary

class VideoNetwork():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.video_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="video_x_video")
        self.audio_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="audio_x_video")
        self.keep_prob = tf.placeholder(dtype, name="keep_prob_video")
        self.learning_rate = tf.placeholder(dtype, name="learning_rate_video")
        self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights_video")
        self.network_type = network_type
        self.VLAD_k = VLAD_K

        x = self.video_input

        if "RVLAD" in network_type.upper():
            NetRVLAD = lp.NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
            x = tf.reshape(x, [-1, 512])
            x = NetRVLAD.forward(x)
        elif "VLAD" == network_type.upper():
            NetVLAD = lp.NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
            x = tf.reshape(x, [-1, 512])
            x = NetVLAD.forward(x)

        x = tf.nn.dropout(x, self.keep_prob)
        x_output = tf.contrib.layers.fully_connected(x, dataset.num_classes, activation_fn=None)

        self.logits = tf.identity(x_output, name="logits_video")

        self.predictions = tf.nn.sigmoid(self.logits, name="predictions_video")

        self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
        self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
        self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
        self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])

        self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_video")

        self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
        self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
        self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
        self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

        self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
            logits=self.logits,
            targets=self.labels,
            pos_weight=self.weights
        )
        self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

        self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
        self._loss = tf.Variable(0.0, trainable=False, name='loss')
        self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
        self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self._batch_loss, colocate_gradients_with_ops=True)

        # AUC PR = mAP
        self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

        self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

        self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

        self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

        self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
        self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
        self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

        # CONFUSION MATRIX
        self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
        self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
        self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
        self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    def initialize(self, sess):
        new_saver = tf.train.Saver()
        new_saver.restore(sess, 'Model/archi-videoResNET_PCA__VGGish_RVLAD64_2020-01-16_17-08-09_model.ckpt')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'confusion_matrix': self._confusion_matrix,
                }

    @property
    def logits_video(self):
        return {
            'logits_video': self.logits
        }

    @property
    def predictions_video(self):
        return {
            'predictions_video': self.predictions
        }

class AudioNetwork():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.video_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="video_x_audio")
        self.audio_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="audio_x_audio")
        self.keep_prob = tf.placeholder(dtype, name="keep_prob_audio")
        self.learning_rate = tf.placeholder(dtype, name="learning_rate_audio")
        self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights_audio")
        self.network_type = network_type
        self.VLAD_k = VLAD_K

        x = self.audio_input

        if "RVLAD" in network_type.upper():
            NetRVLAD = lp.NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
            x = tf.reshape(x, [-1, 512])
            x = NetRVLAD.forward(x)
        elif "VLAD" == network_type.upper():
            NetVLAD = lp.NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
            x = tf.reshape(x, [-1, 512])
            x = NetVLAD.forward(x)

        x = tf.nn.dropout(x, self.keep_prob)
        x_output = tf.contrib.layers.fully_connected(x, dataset.num_classes, activation_fn=None)

        self.logits = tf.identity(x_output, name="logits_audio")

        self.predictions = tf.nn.sigmoid(self.logits, name="predictions_audio")

        self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
        self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
        self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
        self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])

        self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_audio")

        self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
        self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
        self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
        self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

        self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
            logits=self.logits,
            targets=self.labels,
            pos_weight=self.weights
        )
        self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

        self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
        self._loss = tf.Variable(0.0, trainable=False, name='loss')
        self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
        self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self._batch_loss)

        # AUC PR = mAP
        self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

        self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

        self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

        self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

        self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
        self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
        self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

        # CONFUSION MATRIX
        self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
        self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
        self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
        self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    def initialize(self, sess):
        new_saver = tf.train.Saver()
        new_saver.restore(sess, 'Model/archi2ResNET_PCA__VGGish_RVLAD64_2020-01-16_17-08-21_model.ckpt')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'confusion_matrix': self._confusion_matrix,
                }

    @property
    def logits_audio(self):
        return {
            'logits_audio': self.logits
        }

    @property
    def predictions_audio(self):
        return {
            'predictions_audio': self.predictions
        }

class Archi3Prediction():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True, sess=None):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.to_train = False
        self.network_type = network_type
        self.VLAD_k = VLAD_K

        self.video_input_file = "Model/Archi1/logits_video.npy"
        self.audio_input_file = "Model/Archi2/logits_audio.npy"

        with tf.Session() as sess:
            self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights")
            self.video_input = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="logits_video")
            self.audio_input = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="logits_audio")

            self.logits = tf.add(tf.scalar_mul(0.5, self.video_input), tf.scalar_mul(0.5, self.audio_input))

            self.predictions = tf.nn.sigmoid(self.logits, name="predictions_mixte")

            self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
            self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
            self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
            self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])


            self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_mixte")
            self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
            self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
            self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
            self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

            self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
                logits=self.logits,
                targets=self.labels,
                pos_weight=self.weights
            )
            self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

            self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
            self._loss = tf.Variable(0.0, trainable=False, name='loss')
            self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
            self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

            # AUC PR = mAP
            self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

            self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

            self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

            self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

            self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
            self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
            self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

            # CONFUSION MATRIX
            self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
            self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
            self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
            self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'confusion_matrix': self._confusion_matrix,
                }

class AudioVideoArchi4():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.network_type = network_type
        self.VLAD_k = VLAD_K

        self.video_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="x_video")
        self.audio_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="x_audio")

        self.keep_prob = tf.placeholder(dtype, name="keep_prob")
        self.learning_rate = tf.placeholder(dtype, name="learning_rate")
        self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights")

        x_video = self.video_input
        x_audio = self.audio_input

        video_NetRVLAD = lp.NetRVLAD(
            feature_size=512,
            max_samples=dataset.number_frames_in_window,
            cluster_size=int(VLAD_K),
            output_dim=512,
            gating=VLAD_gating,
            add_batch_norm=VLAD_batch_norm,
            is_training=True,
            suffix_tensor_name="_video"
        )

        audio_NetRVLAD = lp.NetRVLAD(
            feature_size=512,
            max_samples=dataset.number_frames_in_window,
            cluster_size=int(VLAD_K),
            output_dim=512,
            gating=VLAD_gating,
            add_batch_norm=VLAD_batch_norm,
            is_training=True,
            suffix_tensor_name="_audio"
        )

        x_video = tf.reshape(x_video, [-1, 512])
        x_video = video_NetRVLAD.forward(x_video)
        x_video = tf.nn.dropout(x_video, self.keep_prob)
        x_video_output = tf.contrib.layers.fully_connected(x_video, dataset.num_classes, activation_fn=None)
        self.logits_video = tf.identity(x_video_output, name="logits_video")

        x_audio = tf.reshape(x_audio, [-1, 512])
        x_audio = audio_NetRVLAD.forward(x_audio)
        x_audio = tf.nn.dropout(x_audio, self.keep_prob)
        x_audio_output = tf.contrib.layers.fully_connected(x_audio, dataset.num_classes, activation_fn=None)
        self.logits_audio = tf.identity(x_audio_output, name="logits_audio")

        self.logits = tf.add(tf.scalar_mul(0.5, self.logits_video), tf.scalar_mul(0.5, self.logits_audio))

        self.predictions = tf.nn.sigmoid(self.logits, name="predictions_audio")

        self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
        self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
        self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
        self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])

        self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_audio")

        self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
        self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
        self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
        self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

        self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
            logits=self.logits,
            targets=self.labels,
            pos_weight=self.weights
        )
        self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

        self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
        self._loss = tf.Variable(0.0, trainable=False, name='loss')
        self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
        self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self._batch_loss)

        # AUC PR = mAP
        self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

        self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

        self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

        self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

        self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
        self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
        self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

        # CONFUSION MATRIX
        self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
        self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
        self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
        self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'confusion_matrix': self._confusion_matrix,
                }

class AudioVideoArchi5():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.network_type = network_type
        self.VLAD_k = VLAD_K

        self.video_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="x_video")
        self.audio_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="x_audio")

        self.keep_prob = tf.placeholder(dtype, name="keep_prob")
        self.learning_rate = tf.placeholder(dtype, name="learning_rate")
        self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights")

        x_video = self.video_input
        x_audio = self.audio_input

        if "RVLAD" in network_type.upper():
            video_NetRVLAD = lp.NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )

            audio_NetRVLAD = lp.NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )

            x_video = tf.reshape(x_video, [-1, 512])
            x_video = video_NetRVLAD.forward(x_video)

            x_audio = tf.reshape(x_audio, [-1, 512])
            x_audio = audio_NetRVLAD.forward(x_audio)
        elif "VLAD" == network_type.upper():
            video_NetVLAD = lp.NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
            x_video = tf.reshape(x_video, [-1, 512])
            x_video = video_NetVLAD.forward(x_video)

            audio_NetVLAD = lp.NetVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
            x_audio = tf.reshape(x_audio, [-1, 512])
            x_audio = audio_NetVLAD.forward(x_audio)



        x_video = tf.nn.dropout(x_video, self.keep_prob)
        x_audio = tf.nn.dropout(x_audio, self.keep_prob)

        x = tf.concat([x_video, x_audio], 1)

        x_output = tf.contrib.layers.fully_connected(x, dataset.num_classes, activation_fn=None)
        self.logits = tf.identity(x_output, name="logits")

        self.predictions = tf.nn.sigmoid(self.logits, name="predictions_audio")

        self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
        self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
        self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
        self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])

        self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_audio")

        self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
        self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
        self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
        self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

        self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
            logits=self.logits,
            targets=self.labels,
            pos_weight=self.weights
        )
        self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

        self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
        self._loss = tf.Variable(0.0, trainable=False, name='loss')
        self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
        self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self._batch_loss)

        # AUC PR = mAP
        self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

        self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

        self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

        self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

        self.pr_curve_0, self.pr_curve_update_op_0 = summary.streaming_op('pr_curve_0_background', labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, metrics_collections='pr')
        self.pr_curve_1, self.pr_curve_update_op_1 = summary.streaming_op('pr_curve_1_cards', labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, metrics_collections='pr')
        self.pr_curve_2, self.pr_curve_update_op_2 = summary.streaming_op('pr_curve_2_subs', labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, metrics_collections='pr')
        self.pr_curve_3, self.pr_curve_update_op_3 = summary.streaming_op('pr_curve_3_goals', labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, metrics_collections='pr')

        self.pr_curve_mean, self.pr_curve_update_op_mean = summary.streaming_op('pr_curve_mean', labels=self.labels, predictions=self.predictions, num_thresholds=200, metrics_collections='pr')

        self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
        self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
        self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

        # CONFUSION MATRIX
        self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
        self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
        self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
        self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    def initialize(self, sess):
        new_saver = tf.train.Saver()
        # new_saver.restore(sess, 'Model/archi5ResNET_PCA__VGGish_RVLAD64_2020-01-23_06-46-15_model.ckpt')
        # new_saver.restore(sess, 'Model/archi14ResNET_PCA__VGGish_RVLAD64_2020-01-28_11-41-02_model.ckpt')
        new_saver.restore(sess, 'TrainVlad/vlad-archi5-20secResNET_PCA__VGGish_VLAD512_2020-03-03_13-30-06_model.ckpt')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'pr_curve_0': self.pr_curve_update_op_0,
                'pr_curve_1': self.pr_curve_update_op_1,
                'pr_curve_2': self.pr_curve_update_op_2,
                'pr_curve_3': self.pr_curve_update_op_3,
                'pr_curve_mean': self.pr_curve_update_op_mean,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'pr_curve_0': self.pr_curve_0,
                'pr_curve_1': self.pr_curve_1,
                'pr_curve_2': self.pr_curve_2,
                'pr_curve_3': self.pr_curve_3,
                'pr_curve_mean': self.pr_curve_mean,
                'confusion_matrix': self._confusion_matrix,
                }

    @property
    def logits_video(self):
        return {
            'logits_video': self.logits
        }

    @property
    def predictions_video(self):
        return {
            'predictions_video': self.predictions
        }

class AudioVideoArchi6():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.network_type = network_type
        self.VLAD_k = VLAD_K

        self.video_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="x_video")
        self.audio_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="x_audio")

        self.keep_prob = tf.placeholder(dtype, name="keep_prob")
        self.learning_rate = tf.placeholder(dtype, name="learning_rate")
        self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights")

        x_video = self.video_input
        x_audio = self.audio_input

        video_NetRVLAD = lp.NetRVLAD(
            feature_size=512,
            max_samples=dataset.number_frames_in_window,
            cluster_size=int(VLAD_K),
            output_dim=512,
            gating=VLAD_gating,
            add_batch_norm=VLAD_batch_norm,
            is_training=True,
            suffix_tensor_name="_video"
        )

        audio_NetRVLAD = lp.NetRVLAD(
            feature_size=512,
            max_samples=dataset.number_frames_in_window,
            cluster_size=int(VLAD_K),
            output_dim=512,
            gating=VLAD_gating,
            add_batch_norm=VLAD_batch_norm,
            is_training=True,
            suffix_tensor_name="_audio"
        )

        x_video = tf.reshape(x_video, [-1, 512])
        x_video = video_NetRVLAD.forward(x_video)

        x_audio = tf.reshape(x_audio, [-1, 512])
        x_audio = audio_NetRVLAD.forward(x_audio)

        x = tf.concat([x_video, x_audio], 1)

        x = tf.nn.dropout(x, self.keep_prob)

        x_output = tf.contrib.layers.fully_connected(x, dataset.num_classes, activation_fn=None)
        self.logits = tf.identity(x_output, name="logits")

        self.predictions = tf.nn.sigmoid(self.logits, name="predictions_audio")

        self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
        self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
        self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
        self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])

        self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_audio")

        self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
        self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
        self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
        self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

        self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
            logits=self.logits,
            targets=self.labels,
            pos_weight=self.weights
        )
        self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

        self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
        self._loss = tf.Variable(0.0, trainable=False, name='loss')
        self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
        self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self._batch_loss)

        # AUC PR = mAP
        self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

        self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

        self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

        self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

        self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
        self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
        self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

        # CONFUSION MATRIX
        self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
        self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
        self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
        self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'confusion_matrix': self._confusion_matrix,
                }

class AudioVideoArchi7():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.network_type = network_type
        self.VLAD_k = VLAD_K

        self.video_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="x_video")
        self.audio_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="x_audio")

        self.keep_prob = tf.placeholder(dtype, name="keep_prob")
        self.learning_rate = tf.placeholder(dtype, name="learning_rate")
        self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights")

        x_video = self.video_input
        x_audio = self.audio_input

        x = tf.concat([x_video, x_audio], 1)

        NetRVLAD = lp.NetRVLAD(
            feature_size=1024,
            max_samples=dataset.number_frames_in_window,
            cluster_size=int(VLAD_K),
            output_dim=1024,
            gating=VLAD_gating,
            add_batch_norm=VLAD_batch_norm,
            is_training=True,
            suffix_tensor_name="_video"
        )

        x = tf.reshape(x, [-1, 1024])
        x = NetRVLAD.forward(x)

        x = tf.nn.dropout(x, self.keep_prob)

        x_output = tf.contrib.layers.fully_connected(x, dataset.num_classes, activation_fn=None)
        self.logits = tf.identity(x_output, name="logits")

        self.predictions = tf.nn.sigmoid(self.logits, name="predictions_audio")

        self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
        self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
        self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
        self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])

        self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_audio")

        self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
        self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
        self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
        self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

        self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
            logits=self.logits,
            targets=self.labels,
            pos_weight=self.weights
        )
        self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

        self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
        self._loss = tf.Variable(0.0, trainable=False, name='loss')
        self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
        self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self._batch_loss)

        # AUC PR = mAP
        self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

        self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

        self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

        self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

        self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
        self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
        self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

        # CONFUSION MATRIX
        self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
        self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
        self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
        self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'confusion_matrix': self._confusion_matrix,
                }

class AudioVideoArchi8():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.network_type = network_type
        self.VLAD_k = VLAD_K

        self.video_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="x_video")
        self.audio_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="x_audio")

        self.keep_prob = tf.placeholder(dtype, name="keep_prob")
        self.learning_rate = tf.placeholder(dtype, name="learning_rate")
        self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights")

        x_video = self.video_input
        x_audio = self.audio_input

        x = tf.concat([x_video, x_audio], 1)

        NetRVLAD = lp.NetRVLAD(
            feature_size=1024,
            max_samples=dataset.number_frames_in_window,
            cluster_size=int(VLAD_K),
            output_dim=512,
            gating=VLAD_gating,
            add_batch_norm=VLAD_batch_norm,
            is_training=True,
            suffix_tensor_name="_video"
        )

        x = tf.reshape(x, [-1, 1024])
        x = NetRVLAD.forward(x)

        x = tf.nn.dropout(x, self.keep_prob)

        x_output = tf.contrib.layers.fully_connected(x, dataset.num_classes, activation_fn=None)
        self.logits = tf.identity(x_output, name="logits")

        self.predictions = tf.nn.sigmoid(self.logits, name="predictions_audio")

        self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
        self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
        self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
        self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])

        self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_audio")

        self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
        self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
        self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
        self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

        self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
            logits=self.logits,
            targets=self.labels,
            pos_weight=self.weights
        )
        self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

        self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
        self._loss = tf.Variable(0.0, trainable=False, name='loss')
        self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
        self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self._batch_loss)

        # AUC PR = mAP
        self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

        self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

        self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

        self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

        self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
        self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
        self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

        # CONFUSION MATRIX
        self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
        self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
        self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
        self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'confusion_matrix': self._confusion_matrix,
                }

class Archi9Prediction():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True, sess=None):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.to_train = False
        self.network_type = network_type
        self.VLAD_k = VLAD_K

        self.video_input_file = "Model/Archi1/logits_video.npy"
        self.audio_input_file = "Model/Archi2/logits_audio.npy"

        with tf.Session() as sess:
            self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights")
            self.video_input = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="logits_video")
            self.audio_input = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="logits_audio")

            self.video_predictions = tf.nn.sigmoid(self.video_input, name="predictions_video")
            self.audio_predictions = tf.nn.sigmoid(self.audio_input, name="predictions_audio")

            predictions = self.video_predictions * self.audio_predictions
            self.predictions = tf.identity(predictions, name="predictions_mixte")

            self.logits = tf.add(tf.scalar_mul(0.5, self.video_input), tf.scalar_mul(0.5, self.audio_input))

            self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
            self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
            self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
            self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])


            self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_mixte")
            self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
            self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
            self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
            self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

            self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
                logits=self.logits,
                targets=self.labels,
                pos_weight=self.weights
            )
            self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

            self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
            self._loss = tf.Variable(0.0, trainable=False, name='loss')
            self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
            self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

            # AUC PR = mAP
            self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

            self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

            self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

            self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

            self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
            self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
            self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

            # CONFUSION MATRIX
            self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
            self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
            self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
            self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'confusion_matrix': self._confusion_matrix,
                'audio_pred': self.audio_predictions,
                'video_pred': self.video_predictions,
                'pred': self.predictions
                }

class AudioVideoArchi10():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.video_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="video_x_video")
        self.audio_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="audio_x_video")
        self.keep_prob = tf.placeholder(dtype, name="keep_prob_video")
        self.learning_rate = tf.placeholder(dtype, name="learning_rate_video")
        self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights_video")
        self.network_type = network_type
        self.VLAD_k = VLAD_K

        x = self.video_input

        if "RVLAD" in network_type.upper():
            NetRVLAD = lp.NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_video"
            )
            x = tf.reshape(x, [-1, 512])
            x = NetRVLAD.forward(x)

        x = tf.nn.dropout(x, self.keep_prob)
        x_output = tf.contrib.layers.fully_connected(x, dataset.num_classes, activation_fn=None)

        self.logits = tf.identity(x_output, name="logits_video")

        self.predictions = tf.nn.sigmoid(self.logits, name="predictions_video")

        self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
        self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
        self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
        self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])

        self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_video")

        self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
        self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
        self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
        self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

        self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
            logits=self.logits,
            targets=self.labels,
            pos_weight=self.weights
        )
        self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

        self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
        self._loss = tf.Variable(0.0, trainable=False, name='loss')
        self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
        self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self._batch_loss)

        # AUC PR = mAP
        self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

        self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

        self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

        self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

        self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
        self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
        self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

        # CONFUSION MATRIX
        self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
        self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
        self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
        self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    def initialize(self, sess):
        new_saver = tf.train.Saver()
        new_saver.restore(sess, 'Model/archi10ResNET_PCA__VGGish_RVLAD64_2020-01-27_13-49-52_model.ckpt')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'confusion_matrix': self._confusion_matrix,
                }

    @property
    def logits_video(self):
        return {
            'logits_video': self.logits
        }

    @property
    def predictions_video(self):
        return {
            'predictions_video': self.predictions
        }

class AudioVideoArchi11():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.video_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="video_x_audio")
        self.audio_input = tf.placeholder(dtype, shape=(None, dataset.number_frames_in_window, 512), name="audio_x_audio")
        self.keep_prob = tf.placeholder(dtype, name="keep_prob_audio")
        self.learning_rate = tf.placeholder(dtype, name="learning_rate_audio")
        self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights_audio")
        self.network_type = network_type
        self.VLAD_k = VLAD_K

        x = self.audio_input

        if "RVLAD" in network_type.upper():
            NetRVLAD = lp.NetRVLAD(
                feature_size=512,
                max_samples=dataset.number_frames_in_window,
                cluster_size=int(VLAD_K),
                output_dim=512,
                gating=VLAD_gating,
                add_batch_norm=VLAD_batch_norm,
                is_training=True,
                suffix_tensor_name="_audio"
            )
            x = tf.reshape(x, [-1, 512])
            x = NetRVLAD.forward(x)

        x = tf.nn.dropout(x, self.keep_prob)
        x_output = tf.contrib.layers.fully_connected(x, dataset.num_classes, activation_fn=None)

        self.logits = tf.identity(x_output, name="logits_audio")

        self.predictions = tf.nn.sigmoid(self.logits, name="predictions_audio")

        self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
        self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
        self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
        self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])

        self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_audio")

        self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
        self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
        self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
        self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

        self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
            logits=self.logits,
            targets=self.labels,
            pos_weight=self.weights
        )
        self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

        self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
        self._loss = tf.Variable(0.0, trainable=False, name='loss')
        self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
        self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self._batch_loss)

        # AUC PR = mAP
        self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

        self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

        self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

        self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

        self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
        self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
        self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

        # CONFUSION MATRIX
        self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
        self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
        self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
        self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    def initialize(self, sess):
        new_saver = tf.train.Saver()
        new_saver.restore(sess, 'Model/archi11ResNET_PCA__VGGish_RVLAD64_2020-01-27_17-04-57_model.ckpt')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'confusion_matrix': self._confusion_matrix,
                }

    @property
    def logits_audio(self):
        return {
            'logits_audio': self.logits
        }

    @property
    def predictions_audio(self):
        return {
            'predictions_audio': self.predictions
        }

class Archi12Prediction():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True, sess=None):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.to_train = False
        self.network_type = network_type
        self.VLAD_k = VLAD_K

        self.video_input_file = "Model/Archi10/logits_video.npy"
        self.audio_input_file = "Model/Archi11/logits_audio.npy"

        with tf.Session() as sess:
            self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights")
            self.video_input = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="logits_video")
            self.audio_input = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="logits_audio")

            self.logits = tf.add(tf.scalar_mul(0.5, self.video_input), tf.scalar_mul(0.5, self.audio_input))

            self.predictions = tf.nn.sigmoid(self.logits, name="predictions_mixte")

            self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
            self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
            self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
            self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])


            self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_mixte")
            self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
            self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
            self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
            self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

            self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
                logits=self.logits,
                targets=self.labels,
                pos_weight=self.weights
            )
            self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

            self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
            self._loss = tf.Variable(0.0, trainable=False, name='loss')
            self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
            self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

            # AUC PR = mAP
            self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

            self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

            self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

            self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

            self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
            self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
            self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

            # CONFUSION MATRIX
            self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
            self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
            self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
            self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'confusion_matrix': self._confusion_matrix,
                }

class Archi18Prediction():
    def __init__(self, dataset, network_type="RVLAD", VLAD_K=64, VLAD_gating=True, VLAD_batch_norm=True, sess=None):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        dtype = tf.float32

        self.to_train = False
        self.network_type = network_type
        self.VLAD_k = VLAD_K

        self.video_input_file = "Model/Archi10/logits_video.npy"
        self.audio_input_file = "Model/Archi11/logits_audio.npy"

        with tf.Session() as sess:
            self.weights = tf.placeholder(dtype, shape=(dataset.num_classes), name="weights")
            self.video_input = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="logits_video")
            self.audio_input = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="logits_audio")

            self.video_predictions = tf.nn.sigmoid(self.video_input, name="predictions_video")
            self.audio_predictions = tf.nn.sigmoid(self.audio_input, name="predictions_audio")

            predictions = self.video_predictions * self.audio_predictions
            self.predictions = tf.identity(predictions, name="predictions_mixte")

            self.logits = tf.add(tf.scalar_mul(0.5, self.video_input), tf.scalar_mul(0.5, self.audio_input))

            self.predictions_0 = tf.transpose(tf.transpose(self.predictions)[0])
            self.predictions_1 = tf.transpose(tf.transpose(self.predictions)[1])
            self.predictions_2 = tf.transpose(tf.transpose(self.predictions)[2])
            self.predictions_3 = tf.transpose(tf.transpose(self.predictions)[3])


            self.labels = tf.placeholder(dtype, shape=(None, dataset.num_classes), name="y_mixte")
            self.labels_0 = tf.transpose(tf.transpose(self.labels)[0])
            self.labels_1 = tf.transpose(tf.transpose(self.labels)[1])
            self.labels_2 = tf.transpose(tf.transpose(self.labels)[2])
            self.labels_3 = tf.transpose(tf.transpose(self.labels)[3])

            self.cross_entropies = tf.nn.weighted_cross_entropy_with_logits(
                logits=self.logits,
                targets=self.labels,
                pos_weight=self.weights
            )
            self.cross_entropy = tf.reduce_sum(self.cross_entropies, axis=1)

            self._batch_loss = tf.reduce_mean(self.cross_entropy, name='batch_loss')
            self._loss = tf.Variable(0.0, trainable=False, name='loss')
            self._loss_update = tf.assign(self._loss, self._loss + self._batch_loss, name='loss_update')
            self._reset_loss_op = tf.assign(self._loss, 0.0, name='reset_loss_op')

            # AUC PR = mAP
            self._auc_PR_0, self._auc_PR_update_0 = tf.metrics.auc(labels=self.labels_0, predictions=self.predictions_0, num_thresholds=200, curve='PR', name='auc_PR_0', )

            self._auc_PR_1, self._auc_PR_update_1 = tf.metrics.auc(labels=self.labels_1, predictions=self.predictions_1, num_thresholds=200, curve='PR', name='auc_PR_1', )

            self._auc_PR_2, self._auc_PR_update_2 = tf.metrics.auc(labels=self.labels_2, predictions=self.predictions_2, num_thresholds=200, curve='PR', name='auc_PR_2', )

            self._auc_PR_3, self._auc_PR_update_3 = tf.metrics.auc(labels=self.labels_3, predictions=self.predictions_3, num_thresholds=200, curve='PR', name='auc_PR_3', )

            self._batch_auc_PR = tf.reduce_mean([self._auc_PR_update_1, self._auc_PR_update_2, self._auc_PR_update_3], name='batch_auc_PR')
            self._auc_PR = tf.Variable(0.0, trainable=False, name='auc_PR')
            self._auc_PR_update = tf.assign(self._auc_PR, self._batch_auc_PR, name='auc_PR_update')

            # CONFUSION MATRIX
            self._batch_confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1), num_classes=dataset.num_classes, name='batch_confusion_matrix')
            self._confusion_matrix = tf.Variable(np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), trainable=False, name='confusion_matrix')
            self._confusion_matrix_update = tf.assign(self._confusion_matrix, self._confusion_matrix + self._batch_confusion_matrix, name='confusion_matrix_update')
            self._reset_confusion_matrix_op = tf.assign(self._confusion_matrix, np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32), name='reset_confusion_matrix_op')

    @property
    def loss(self):
        return self._loss

    @property
    def update_metrics_op(self):
        return {'confusion_matrix': self._confusion_matrix_update,
                'auc_PR': self._auc_PR_update,
                'auc_PR_0': self._auc_PR_update_0,
                'auc_PR_1': self._auc_PR_update_1,
                'auc_PR_2': self._auc_PR_update_2,
                'auc_PR_3': self._auc_PR_update_3,
                'loss': self._loss_update,
                }

    @property
    def reset_metrics_op(self):
        return {'confusion_matrix': self._reset_confusion_matrix_op,
                'loss': self._reset_loss_op,
                }

    @property
    def metrics_op(self):
        return {'loss': self._loss,
                'auc_PR': self._auc_PR,
                'auc_PR_0': self._auc_PR_0,
                'auc_PR_1': self._auc_PR_1,
                'auc_PR_2': self._auc_PR_2,
                'auc_PR_3': self._auc_PR_3,
                'confusion_matrix': self._confusion_matrix,
                'audio_pred': self.audio_predictions,
                'video_pred': self.video_predictions,
                'pred': self.predictions
                }
