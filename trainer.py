import numpy as np
import os
import tensorflow as tf
import time

from datetime import date

class Trainer():
    def __init__(self, dataset, network, video_network=None, audio_network=None, output_prefix=""):
        self.network = network
        self.dataset = dataset
        self.output_prefix = output_prefix + "_" +str(date.today().isoformat() + time.strftime('_%H-%M-%S'))

    def train(self, epochs=1, learning_rate=0.001, tflog="logs"):
        self.tflog = tflog
        config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=config) as sess:
            # Initializing the variables
            sess.run(tf.global_variables_initializer())

            # Define writer
            saver = tf.train.Saver()
            if not os.path.exists(self.tflog):
                os.makedirs(self.tflog)
            self.train_writer = tf.summary.FileWriter(os.path.join(self.tflog, self.output_prefix + "_training"), sess.graph)
            self.valid_writer = tf.summary.FileWriter(os.path.join(self.tflog, self.output_prefix + "_validation"), sess.graph)
            self.test_writer  = tf.summary.FileWriter(os.path.join(self.tflog, self.output_prefix + "_testing"), sess.graph)

            best_validation_mAP = 0
            cnt_since_best_epoch = 0

            for epoch in range(epochs):
                start_time_epoch = time.time()

                print("\n\n\n")
                print("Epoch {:>2}: ".format(epoch + 1))

                self.dataset.prepareNewEpoch()

                print("\nTraining")

                sess.run([self.network.reset_metrics_op])

                sess.run(tf.local_variables_initializer())

                start_time = time.time()

                for total_num_batches in range(self.dataset.nb_batch_training):
                    batch_video_features, batch_audio_features, batch_labels, batch_indices = self.dataset.getTrainingBatch(total_num_batches)
                    feed_dict = {
                        self.network.video_input: batch_video_features,
                        self.network.audio_input: batch_audio_features,
                        self.network.labels: batch_labels,
                        self.network.keep_prob: 0.6,
                        self.network.learning_rate: learning_rate,
                        self.network.weights: self.dataset.weights
                    }

                    sess.run(self.network.optimizer, feed_dict=feed_dict) # optimize
                    sess.run(self.network.update_metrics_op, feed_dict=feed_dict) # update metrics
                    vals_train = sess.run(self.network.metrics_op, feed_dict=feed_dict) # return metrics
                    total_num_batches += 1

                    good_sample = np.sum(np.multiply(vals_train['confusion_matrix'], np.identity(4)), axis=0)
                    bad_sample = np.sum(vals_train['confusion_matrix'] - np.multiply(vals_train['confusion_matrix'], np.identity(4)), axis=0)
                    vals_train['accuracies'] = good_sample / (bad_sample + good_sample)
                    vals_train['accuracy'] = np.mean(vals_train['accuracies'])
                    vals_train['mAP'] = np.mean([vals_train['auc_PR_1'], vals_train['auc_PR_2'], vals_train['auc_PR_3']])

                    print(("Batch number: %.3f Loss: %.3f Accuracy: %.3f mAP: %.3f") % (total_num_batches, vals_train['loss'], vals_train['accuracy'], vals_train['mAP']))
                    print(("auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)") %
                    (vals_train['auc_PR'], vals_train['auc_PR_0'], vals_train['auc_PR_1'], vals_train['auc_PR_2'], vals_train['auc_PR_3']))

                print(vals_train['confusion_matrix'])

                print(" Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}".format(vals_train['loss'], vals_train['accuracy'], vals_train['mAP']))
                print(" Time: {:<8.3} s".format(time.time()-start_time), flush=True)

                summaries = [
                    tf.Summary.Value(tag="learning_rate",       simple_value=learning_rate),
                    tf.Summary.Value(tag="loss",                simple_value=vals_train['loss']),
                    tf.Summary.Value(tag="accuracy/average",    simple_value=vals_train['accuracy']),
                    tf.Summary.Value(tag="accuracy/0_background",   simple_value=vals_train['accuracies'][0]),
                    tf.Summary.Value(tag="accuracy/1_cards",    simple_value=vals_train['accuracies'][1]),
                    tf.Summary.Value(tag="accuracy/2_subs",     simple_value=vals_train['accuracies'][2]),
                    tf.Summary.Value(tag="accuracy/3_goals",    simple_value=vals_train['accuracies'][3]),
                    tf.Summary.Value(tag="AP/mean",             simple_value=vals_train['mAP']),
                    tf.Summary.Value(tag="AP/0_background",     simple_value=vals_train['auc_PR_0']),
                    tf.Summary.Value(tag="AP/1_cards",          simple_value=vals_train['auc_PR_1']),
                    tf.Summary.Value(tag="AP/2_subs",           simple_value=vals_train['auc_PR_2']),
                    tf.Summary.Value(tag="AP/3_goals",          simple_value=vals_train['auc_PR_3']),
                ]
                self.train_writer.add_summary(tf.Summary(value=summaries), epoch)

                # Run Multiple Validation to check the metrics are constant
                print("\n")
                print("Validation")

                vals_valid = self.validate(sess)

                summaries = [
                    tf.Summary.Value(tag="learning_rate",       simple_value=learning_rate),
                    tf.Summary.Value(tag="loss",                simple_value=vals_valid['loss']),
                    tf.Summary.Value(tag="accuracy/average",    simple_value=vals_valid['accuracy']),
                    tf.Summary.Value(tag="accuracy/0_background",   simple_value=vals_valid['accuracies'][0]),
                    tf.Summary.Value(tag="accuracy/1_cards",    simple_value=vals_valid['accuracies'][1]),
                    tf.Summary.Value(tag="accuracy/2_subs",     simple_value=vals_valid['accuracies'][2]),
                    tf.Summary.Value(tag="accuracy/3_goals",    simple_value=vals_valid['accuracies'][3]),
                    tf.Summary.Value(tag="AP/mean",             simple_value=vals_valid['mAP']),
                    tf.Summary.Value(tag="AP/0_background",     simple_value=vals_valid['auc_PR_0']),
                    tf.Summary.Value(tag="AP/1_cards",          simple_value=vals_valid['auc_PR_1']),
                    tf.Summary.Value(tag="AP/2_subs",           simple_value=vals_valid['auc_PR_2']),
                    tf.Summary.Value(tag="AP/3_goals",          simple_value=vals_valid['auc_PR_3']),
                ]
                self.valid_writer.add_summary(tf.Summary(value=summaries), epoch)

                # Look for best model
                print("\n")
                print("validation_mAP: " + str(vals_valid['mAP']))
                print("best_validation_mAP: " + str(best_validation_mAP))
                print("validation_mAP > best_validation_mAP ?: " + str(vals_valid['mAP'] > best_validation_mAP))
                print("cnt_since_best_epoch currently: " + str(cnt_since_best_epoch))
                print("elapsed time for this epoch: " + str(time.time() - start_time_epoch))
                if(vals_valid['mAP'] > best_validation_mAP):
                    best_validation_mAP = vals_valid['mAP']
                    best_validation_accuracy = vals_valid["accuracy"]
                    best_validation_loss = vals_valid['loss']
                    best_epoch = epoch
                    cnt_since_best_epoch = 0
                    best_output_prefix = self.output_prefix
                    saver.save(sess, os.path.join(self.tflog, best_output_prefix + "_model.ckpt"))
                else:
                    cnt_since_best_epoch += 1

                if cnt_since_best_epoch > 10 and learning_rate > 0.0001:
                    print("reducing LR after plateau since " + str(cnt_since_best_epoch) + " epochs without improvements")
                    learning_rate /= 10
                    cnt_since_best_epoch = 0
                    saver.restore(sess, os.path.join(self.tflog, self.output_prefix + "_model.ckpt"))
                elif cnt_since_best_epoch > 30:
                    print("stopping after plateau since " + str(cnt_since_best_epoch) + " epochs without improvements")
                    break

            self.train_writer.close()
            self.valid_writer.close()
            print("stopping after " + str(epoch) + " epochs maximum training reached")

            print("\nTesting")
            saver.restore(sess, os.path.join(self.tflog, best_output_prefix + "_model.ckpt"))

            vals_test = self.test(sess)

            summaries = [
                tf.Summary.Value(tag="learning_rate",       simple_value=learning_rate),
                tf.Summary.Value(tag="loss",                simple_value=vals_test['loss']),
                tf.Summary.Value(tag="accuracy/average",    simple_value=vals_test['accuracy']),
                tf.Summary.Value(tag="accuracy/0_background",   simple_value=vals_test['accuracies'][0]),
                tf.Summary.Value(tag="accuracy/1_cards",    simple_value=vals_test['accuracies'][1]),
                tf.Summary.Value(tag="accuracy/2_subs",     simple_value=vals_test['accuracies'][2]),
                tf.Summary.Value(tag="accuracy/3_goals",    simple_value=vals_test['accuracies'][3]),
                tf.Summary.Value(tag="AP/mean",             simple_value=vals_test['mAP']),
                tf.Summary.Value(tag="AP/0_background",     simple_value=vals_test['auc_PR_0']),
                tf.Summary.Value(tag="AP/1_cards",          simple_value=vals_test['auc_PR_1']),
                tf.Summary.Value(tag="AP/2_subs",           simple_value=vals_test['auc_PR_2']),
                tf.Summary.Value(tag="AP/3_goals",          simple_value=vals_test['auc_PR_3'])
            ]
            self.test_writer.add_summary(tf.Summary(value=summaries), epoch)
            self.test_writer.close()

        return vals_train, vals_valid, vals_test, best_output_prefix

    def validate(self, sess):
        sess.run(tf.local_variables_initializer())
        sess.run([self.network.reset_metrics_op])

        start_time = time.time()
        total_num_batches = 0

        for i in range(self.dataset.nb_batch_validation):
            batch_video_features, batch_audio_features, batch_labels = self.dataset.getValidationBatch(i)

            feed_dict = {
                self.network.video_input: batch_video_features,
                self.network.audio_input: batch_audio_features,
                self.network.labels: batch_labels,
                self.network.keep_prob: 1.0,
                self.network.weights: self.dataset.weights
            }

            sess.run(self.network.update_metrics_op, feed_dict=feed_dict)
            vals_valid = sess.run(self.network.metrics_op, feed_dict=feed_dict)

            total_num_batches += 1

            vals_valid['mAP'] = np.mean([vals_valid['auc_PR_1'], vals_valid['auc_PR_2'], vals_valid['auc_PR_3']])

        good_sample = np.sum( np.multiply(vals_valid['confusion_matrix'], np.identity(4)), axis=0)
        bad_sample = np.sum( vals_valid['confusion_matrix'] - np.multiply(vals_valid['confusion_matrix'], np.identity(4)), axis=0)
        vals_valid['accuracies'] =  good_sample / ( bad_sample + good_sample )
        vals_valid['accuracy'] = np.mean(vals_valid['accuracies'])

        print(vals_valid['confusion_matrix'])
        print(("auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)") %
        (vals_valid['auc_PR'], vals_valid['auc_PR_0'], vals_valid['auc_PR_1'], vals_valid['auc_PR_2'], vals_valid['auc_PR_3']))
        print(" Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}".format(vals_valid['loss'], vals_valid['accuracy'], vals_valid['mAP']))
        print(" Time: {:<8.3} s".format(time.time()-start_time))

        return vals_valid

    def test(self, sess):
        sess.run(tf.local_variables_initializer())
        sess.run([self.network.reset_metrics_op])

        start_time = time.time()
        total_num_batches = 0
        for i in range(self.dataset.nb_batch_testing):
            batch_video_features, batch_audio_features, batch_labels = self.dataset.getTestingBatch(i)

            feed_dict = {
                self.network.video_input: batch_video_features,
                self.network.audio_input: batch_audio_features,
                self.network.labels: batch_labels,
                self.network.keep_prob: 1.0,
                self.network.weights: self.dataset.weights
            }

            sess.run([self.network.loss], feed_dict=feed_dict)
            sess.run(self.network.update_metrics_op, feed_dict=feed_dict)
            vals_test = sess.run(self.network.metrics_op, feed_dict=feed_dict)

            total_num_batches += 1

            vals_test['mAP'] = np.mean([vals_test['auc_PR_1'], vals_test['auc_PR_2'], vals_test['auc_PR_3']])

        good_sample = np.sum( np.multiply(vals_test['confusion_matrix'], np.identity(4)), axis=0)
        bad_sample = np.sum( vals_test['confusion_matrix'] - np.multiply(vals_test['confusion_matrix'], np.identity(4)), axis=0)
        vals_test['accuracies'] =  good_sample / ( bad_sample + good_sample )
        vals_test['accuracy'] = np.mean(vals_test['accuracies'])

        print(vals_test['confusion_matrix'])
        print(("auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)") %
        (vals_test['auc_PR'], vals_test['auc_PR_0'], vals_test['auc_PR_1'], vals_test['auc_PR_2'], vals_test['auc_PR_3']))
        print(" Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}".format(vals_test['loss'], vals_test['accuracy'], vals_test['mAP']))
        print(" Time: {:<8.3} s".format(time.time()-start_time))

        return vals_test

    def predict(self, prop, display=True, tflog="logs"):
        self.tflog = tflog
        with tf.Session() as sess:
            if not os.path.exists(self.tflog):
                os.makedirs(self.tflog)
            self.network.initialize(sess)

            sess.run(tf.local_variables_initializer())
            sess.run([self.network.reset_metrics_op])

            start_time = time.time()
            total_num_batches = 0
            prop_l = []
            for i in range(self.dataset.nb_batch_testing):
                batch_video_features, batch_audio_features, batch_labels = self.dataset.getTestingBatch(i)

                feed_dict = {
                    self.network.video_input: batch_video_features,
                    self.network.audio_input: batch_audio_features,
                    self.network.labels: batch_labels,
                    self.network.keep_prob: 1.0,
                    self.network.weights: self.dataset.weights
                }

                sess.run([self.network.loss], feed_dict=feed_dict)
                sess.run(self.network.update_metrics_op, feed_dict=feed_dict)
                vals_test = sess.run(self.network.metrics_op, feed_dict=feed_dict)
                prop_output = sess.run(getattr(self.network, prop), feed_dict=feed_dict)
                prop_l.append(prop_output[prop])
                total_num_batches += 1

                vals_test['mAP'] = np.mean([vals_test['auc_PR_1'], vals_test['auc_PR_2'], vals_test['auc_PR_3']])

            good_sample = np.sum( np.multiply(vals_test['confusion_matrix'], np.identity(4)), axis=0)
            bad_sample = np.sum( vals_test['confusion_matrix'] - np.multiply(vals_test['confusion_matrix'], np.identity(4)), axis=0)
            vals_test['accuracies'] =  good_sample / ( bad_sample + good_sample )
            vals_test['accuracy'] = np.mean(vals_test['accuracies'])

            prop_l = np.array(prop_l)
            np.save(self.tflog + "/" + prop, prop_l)

            if display:
                print(vals_test['confusion_matrix'])
                print(("auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)") %
                (vals_test['auc_PR'], vals_test['auc_PR_0'], vals_test['auc_PR_1'], vals_test['auc_PR_2'], vals_test['auc_PR_3']))
                print(" Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}".format(vals_test['loss'], vals_test['accuracy'], vals_test['mAP']))
                print(" Time: {:<8.3} s".format(time.time()-start_time))

        return vals_test

    def predict_other(self):
        video_logits = np.load(self.network.video_input_file, allow_pickle=True)
        audio_logits = np.load(self.network.audio_input_file, allow_pickle=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run([self.network.reset_metrics_op])
            sess.run(tf.local_variables_initializer())

            start_time = time.time()
            total_num_batches = 0
            prop_l = []
            for i in range(self.dataset.nb_batch_testing):
                _, _, batch_labels = self.dataset.getTestingBatch(i)

                feed_dict = {
                    self.network.video_input: video_logits[i],
                    self.network.audio_input: audio_logits[i],
                    self.network.labels: batch_labels,
                    self.network.weights: self.dataset.weights
                }

                sess.run([self.network.loss], feed_dict=feed_dict)
                sess.run(self.network.update_metrics_op, feed_dict=feed_dict)
                vals_test = sess.run(self.network.metrics_op, feed_dict=feed_dict)
                total_num_batches += 1

                vals_test['mAP'] = np.mean([vals_test['auc_PR_1'], vals_test['auc_PR_2'], vals_test['auc_PR_3']])

            good_sample = np.sum( np.multiply(vals_test['confusion_matrix'], np.identity(4)), axis=0)
            bad_sample = np.sum( vals_test['confusion_matrix'] - np.multiply(vals_test['confusion_matrix'], np.identity(4)), axis=0)
            vals_test['accuracies'] =  good_sample / ( bad_sample + good_sample )
            vals_test['accuracy'] = np.mean(vals_test['accuracies'])

            print(vals_test['confusion_matrix'])
            print(("auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)") %
            (vals_test['auc_PR'], vals_test['auc_PR_0'], vals_test['auc_PR_1'], vals_test['auc_PR_2'], vals_test['auc_PR_3']))
            print(" Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}".format(vals_test['loss'], vals_test['accuracy'], vals_test['mAP']))
            print(" Time: {:<8.3} s".format(time.time()-start_time))
