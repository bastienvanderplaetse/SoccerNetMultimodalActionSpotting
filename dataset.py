import json
import numpy as np
import os
import random

from pprint import pprint

import time

def create_directory(dir_name):
    if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
            os.mkdir(dir_name)

class Dataset():

    def __init__(self, window_size_sec, feature_per_second, batch_size, system, augmentation=False):
        print("Init dataset")

        self.window_size_sec = window_size_sec
        self.feature_per_second = feature_per_second
        self.number_frames_in_window = window_size_sec * feature_per_second
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.system = system

        self.ACTION_DICT = {
            'card': 1,
            'subs': 2,
            'soccer': 3
        }

        self.action_list = ['background', 'card', 'subs', 'soccer']
        self.num_classes = len(self.action_list)
        self.weights = [1] * len(self.action_list)

    def prepareDataset(self, path_data_train, path_data_valid, path_data_test, featureVideoName, featureAudioName, PCA):
        self.batch_dir = "Windows_size_{}_sec".format(self.window_size_sec)
        create_directory(self.batch_dir)

        self.training_GamesKeys = []
        self.training_features = {}
        self.prepare(path_data_train, featureVideoName, featureAudioName, PCA, self.training_GamesKeys, self.training_features, "train")

        self.validation_GamesKeys = []
        self.validation_features = {}
        self.prepare(path_data_valid, featureVideoName, featureAudioName, PCA, self.validation_GamesKeys, self.validation_features, "valid")

        self.testing_GamesKeys = []
        self.testing_features = {}
        self.prepare(path_data_test, featureVideoName, featureAudioName, PCA, self.testing_GamesKeys, self.testing_features, "test")

    def prepare(self, path_data, featureVideoName, featureAudioName, PCA, GamesKeys, features, directory):
        if self.augmentation:
            temp_dir = os.path.join(self.batch_dir, "augmented")
        else:
            temp_dir = os.path.join(self.batch_dir, "normal")
        create_directory(temp_dir)
        directory = os.path.join(temp_dir, directory)
        create_directory(directory)

        path_data_dirname, path_data_basename = os.path.split(path_data)
        listGames = np.load(path_data)

        for gamePath in listGames:
            gamePath = os.path.join(path_data_dirname, gamePath)

            featureFullPath = {"video": ["", ""], "audio": ["", ""], "keys": ["Half_1", "Half_2"]}
            cpt = 0
            for featureFileName in os.listdir(gamePath):
                found = False
                if featureVideoName in featureFileName and "float16" not in featureFileName and ((PCA and "PCA" in featureFileName) or (not PCA and "PCA" not in featureFileName)):
                    featType = "video"
                    if "1_" in featureFileName:
                        half_key = 0
                        found = True
                    elif "2_" in featureFileName:
                        half_key = 1
                        found = True
                elif featureAudioName in featureFileName and "float16" not in featureFileName:
                    featType = "audio"
                    if "1_" in featureFileName:
                        half_key = 0
                        found = True
                    elif "2_" in featureFileName:
                        half_key = 1
                        found = True
                if found:
                    featureFullPath[featType][half_key] = os.path.join(gamePath, featureFileName)
                    cpt += 1
                    if cpt == 4:
                        break

            for i in range(2):
                key = os.path.join(gamePath, featureFullPath['keys'][i])
                if self.system == 'windows':
                    key = key.replace('/', '\\')
                GamesKeys.append(key)
                video_features_cont = np.load(featureFullPath['video'][i])
                audio_features_cont = np.load(featureFullPath['audio'][i])

                if self.augmentation:
                    pass
                else:
                    self.prepare_normal(gamePath, video_features_cont, audio_features_cont, features, key, i+1, directory, featureVideoName, featureAudioName)

    def prepare_normal(self, gamePath, video_features_cont, audio_features_cont, features, key, half, directory, featureVideoName, featureAudioName):
        cnt_data_augmentation = 0

        l = video_features_cont.shape[0] - video_features_cont.shape[0] % self.number_frames_in_window
        features_video = np.zeros((cnt_data_augmentation + int(l/self.number_frames_in_window), self.number_frames_in_window, 512))
        features_audio = np.zeros((cnt_data_augmentation + int(l/self.number_frames_in_window), self.number_frames_in_window, 512))

        cnt_data_augmentation = 0
        for minframe in np.reshape(video_features_cont[0:l,:], (-1, self.number_frames_in_window, 512)):
            features_video[cnt_data_augmentation] = minframe
            cnt_data_augmentation += 1

        cnt_data_augmentation = 0
        for minframe in np.reshape(audio_features_cont[0:l,:], (-1, self.number_frames_in_window, 512)):
            features_audio[cnt_data_augmentation] = minframe
            cnt_data_augmentation += 1

        labelFullPath = os.path.join(gamePath, "Labels.json")
        with open(labelFullPath) as labelFile:
            jsonLabel = json.loads(labelFile.read())

        Labels = np.zeros((features_video.shape[0], len(self.action_list)), dtype=int)
        Labels[:,0] = 1

        for event in jsonLabel['annotations']:
            Time_Half = int(event['gameTime'][0])
            Time_Minute = int(event['gameTime'][-5:-3])
            Time_Second = int(event['gameTime'][-2:])

            if ("card" in event["label"]): label = 1
            elif ("subs" in event["label"]): label = 2
            elif ("soccer" in event["label"]): label = 3
            else: label = 0

            if Time_Half == half:
                index = min(
                    int(60/self.window_size_sec * Time_Minute + int(Time_Second / self.window_size_sec)),
                    Labels.shape[0]-1
                )

                Labels[index, 0] = 0
                Labels[index, label] = 1

        if self.system == 'linux':
            sep = '/'
        else:
            sep = '\\'

        video_filename = os.path.join(directory, "{}_video_{}.npy".format('_'.join(key.split(sep)[-4:]), featureVideoName))
        audio_filename = os.path.join(directory, "{}_audio_{}.npy".format('_'.join(key.split(sep)[-4:]), featureAudioName))
        label_filename = os.path.join(directory, "{}_labels.npy".format('_'.join(key.split(sep)[-4:])))

        np.save(video_filename, features_video, allow_pickle=True)
        np.save(audio_filename, features_audio, allow_pickle=True)
        np.save(label_filename, Labels, allow_pickle=True)

        features[key] = dict()
        features[key]['video'] = video_filename
        features[key]['audio'] = audio_filename
        features[key]['label'] = label_filename

    def loadDataset(self, path_data_train, path_data_valid, path_data_test, featureVideoName, featureAudioName, PCA):
        self.batch_dir = "Windows_size_{}_sec".format(self.window_size_sec)

        self.training_GamesKeys = []
        self.training_features = {}
        self.load(path_data_train, featureVideoName, featureAudioName, PCA, self.training_GamesKeys, self.training_features, "train")

        self.validation_GamesKeys = []
        self.validation_features = {}
        self.load(path_data_valid, featureVideoName, featureAudioName, PCA, self.validation_GamesKeys, self.validation_features, "valid")

        self.testing_GamesKeys = []
        self.testing_features = {}
        self.load(path_data_test, featureVideoName, featureAudioName, PCA, self.testing_GamesKeys, self.testing_features, "test")

    def load(self, path_data, featureVideoName, featureAudioName, PCA, GamesKeys, features, directory):
        if self.augmentation:
            temp_dir = os.path.join(self.batch_dir, "augmented")
        else:
            temp_dir = os.path.join(self.batch_dir, "normal")
        directory = os.path.join(temp_dir, directory)

        if self.augmentation:
            pass
        else:
            self.load_normal(path_data, GamesKeys, features, featureVideoName, featureAudioName, directory)

    def load_normal(self, path_data, GamesKeys, features, featureVideoName, featureAudioName, directory):
        path_data_dirname, path_data_basename = os.path.split(path_data)
        listGames = np.load(path_data)

        for gamePath in listGames:
            gamePath = os.path.join(path_data_dirname, gamePath)

            featureFullPath = {"video": ["", ""], "audio": ["", ""], "keys": ["Half_1", "Half_2"]}

            for i in range(2):
                key = os.path.join(gamePath, featureFullPath['keys'][i])
                if self.system == "windows":
                    key = key.replace('/', '\\')
                GamesKeys.append(key)
                if self.system == "linux":
                    sep = '/'
                else:
                    sep = '\\'

                features[key] = dict()

                video_filename = os.path.join(directory, "{}_video_{}.npy".format('_'.join(key.split(sep)[-4:]), featureVideoName))
                audio_filename = os.path.join(directory, "{}_audio_{}.npy".format('_'.join(key.split(sep)[-4:]), featureAudioName))
                label_filename = os.path.join(directory, "{}_labels.npy".format('_'.join(key.split(sep)[-4:])))

                features[key]['video'] = video_filename
                features[key]['audio'] = audio_filename
                features[key]['label'] = label_filename

    def prepareNewEpoch(self):
        if self.augmentation:
            pass
        else:
            self.prepareNewEpochNormal()

    def prepareNewEpochNormal(self):
        random.shuffle(self.training_GamesKeys)

        nb_halves = len(self.training_GamesKeys)
        print("batch_size:", self.batch_size)
        print("nb_halves:", nb_halves)

        self.nb_batch_training = int(np.ceil(nb_halves/self.batch_size))
        self.nb_batch_validation = len(self.validation_GamesKeys)
        self.nb_batch_testing = len(self.testing_GamesKeys)

        self._current_training_batch_index = -1
        self._current_validation_batch_index = -1

    def getTrainingBatch(self, num_batch):
        if self.augmentation:
            pass
        else:
            return self.getNormalTrainingBatch(num_batch)

    def getNormalTrainingBatch(self, num_batch):
        self._current_training_batch_index = num_batch

        init_games = num_batch * self.batch_size
        end_games = min((num_batch+1)*self.batch_size, len(self.training_GamesKeys))

        return self.getGamesBatch(init_games, end_games)

    def getGamesBatch(self, init_games, end_games):
        key = self.training_GamesKeys[init_games]

        train_batch_video_features = [np.load(self.training_features[key]['video'])]
        train_batch_audio_features = [np.load(self.training_features[key]['audio'])]
        train_batch_labels = [np.load(self.training_features[key]['label'])]

        train_batch_indices = []

        for gameKey in self.training_GamesKeys[init_games+1:end_games]:
            train_batch_video_features.append(np.load(self.training_features[gameKey]['video']))
            train_batch_audio_features.append(np.load(self.training_features[gameKey]['audio']))
            train_batch_labels.append(np.load(self.training_features[gameKey]['label']))

        train_batch_video_features = np.concatenate([row for row in train_batch_video_features])
        train_batch_audio_features = np.concatenate([row for row in train_batch_audio_features])
        train_batch_labels = np.concatenate([row for row in train_batch_labels])

        return train_batch_video_features, train_batch_audio_features, train_batch_labels, train_batch_indices

    def getValidationBatch(self, num_batch):
        valid_batch_features = self.validation_features[self.validation_GamesKeys[num_batch]]
        valid_batch_video_features = np.load(valid_batch_features['video'])
        valid_batch_audio_features = np.load(valid_batch_features['audio'])
        valid_batch_labels = np.load(valid_batch_features['label'])

        self._current_validation_batch_index = num_batch

        return valid_batch_video_features, valid_batch_audio_features, valid_batch_labels

    def getTestingBatch(self, num_batch):
        testing_batch_features = self.testing_features[self.testing_GamesKeys[num_batch]]
        testing_batch_video_features = np.load(testing_batch_features['video'])
        testing_batch_audio_features = np.load(testing_batch_features['audio'])
        testing_batch_labels = np.load(testing_batch_features['label'])

        self._current_testing_batch_index = num_batch

        return testing_batch_video_features, testing_batch_audio_features, testing_batch_labels

    def loadSpottingTestingDataset(self, path_data, featureVideoName, featureAudioName, PCA=True, window_size_sec=60, feature_per_second=2):
        self.window_size_sec = window_size_sec
        self.featurePerSecond = feature_per_second
        self.number_frames_in_window = window_size_sec * feature_per_second
        self.featureVideoName = featureVideoName
        self.featureAudioName = featureAudioName

        self.PCA = PCA

        print("Prepare Action Spotting Testing dataset")
        print(path_data)
        path_data_dirname, path_data_basename = os.path.split(path_data)
        print(path_data_dirname)
        print(path_data_basename)

        listGames = np.load(path_data)
        import sys
        sys.exit(0)
        self.testing_GamesKeys = []
        self.testing_features = {}

        for gamePath in listGames:
            gamePath = os.path.join(path_data_dirname, gamePath)

            featureFullPath = {"video": ["", ""], "audio": ["", ""], "keys": ["Half_1", "Half_2"]}
            cpt = 0
            for featureFileName in os.listdir(gamePath):
                found = False
                if featureVideoName in featureFileName and "float16" not in featureFileName and ((PCA and "PCA" in featureFileName) or (not PCA and "PCA" not in featureFileName)):
                    featType = "video"
                    if "1_" in featureFileName:
                        half_key = 0
                        found = True
                    elif "2_" in featureFileName:
                        half_key = 1
                        found = True
                elif featureAudioName in featureFileName and "float16" not in featureFileName:
                    featType = "audio"
                    if "1_" in featureFileName:
                        half_key = 0
                        found = True
                    elif "2_" in featureFileName:
                        half_key = 1
                        found = True
                if found:
                    featureFullPath[featType][half_key] = os.path.join(gamePath, featureFileName)
                    cpt += 1
                    if cpt == 4:
                        break

            for i in range(2):
                key = os.path.join(gamePath, featureFullPath['keys'][i])
                self.testing_GamesKeys.append(key)
                self.testing_features[key] = {}
                self.testing_features[key]['video'] = featureFullPath['video'][i]
                self.testing_features[key]['audio'] = featureFullPath['audio'][i]
                self.testing_features[key]['label'] = os.path.join(gamePath, "Labels.json")
                self.testing_features[key]['half'] = i



        self.weights = [1, 1, 1, 1]

        self.nb_batch_testing = len(self.testing_GamesKeys)

    def getSpottingTestingBatch(self, num_batch):
        testing_batch_features = self.testing_features[self.testing_GamesKeys[num_batch]]

        # Video
        testing_features_cont = np.load(testing_batch_features['video'])

        strides = testing_features_cont.strides
        nb_frames = testing_features_cont.shape[0]
        size_feature = testing_features_cont.shape[1]
        sliding_window_seconds = self.window_size_sec
        sliding_window_frame = sliding_window_seconds * self.featurePerSecond

        testing_features_cont = np.append([testing_features_cont[0,:]]*sliding_window_seconds, testing_features_cont, axis=0)
        testing_features_cont = np.append(testing_features_cont, [testing_features_cont[-1,:]]*sliding_window_seconds, axis=0)

        testing_batch_video_features = np.lib.stride_tricks.as_strided(testing_features_cont, shape=(int(nb_frames/2), sliding_window_frame, size_feature), strides=(strides[0]*2,strides[0],strides[1]))

        for pl in range(20, 40):
            for pk in range(10):
                assert((testing_batch_video_features[pl,pk,:] - testing_features_cont[pl*2+pk,:]).sum() == 0)

        # Audio
        testing_features_cont = np.load(testing_batch_features['audio'])

        strides = testing_features_cont.strides
        nb_frames = testing_features_cont.shape[0]
        size_feature = testing_features_cont.shape[1]
        sliding_window_seconds = self.window_size_sec
        sliding_window_frame = sliding_window_seconds * self.featurePerSecond

        testing_features_cont = np.append([testing_features_cont[0,:]]*sliding_window_seconds, testing_features_cont, axis=0)
        testing_features_cont = np.append(testing_features_cont, [testing_features_cont[-1,:]]*sliding_window_seconds, axis=0)

        testing_batch_audio_features = np.lib.stride_tricks.as_strided(testing_features_cont, shape=(int(nb_frames/2), sliding_window_frame, size_feature), strides=(strides[0]*2,strides[0],strides[1]))

        for pl in range(20, 40):
            for pk in range(10):
                assert((testing_batch_audio_features[pl,pk,:] - testing_features_cont[pl*2+pk,:]).sum() == 0)

        # Labels
        labelFullPath = testing_batch_features['label']
        with open(labelFullPath) as labelFile:
            jsonLabel = json.loads(labelFile.read())

        Labels = np.zeros((testing_batch_audio_features.shape[0], 4), dtype=int)
        Labels[:,0] = 1

        for event in jsonLabel["annotations"]:
            Half = int(event['gameTime'][0])
            Time_Minute = int(event['gameTime'][-5:-3])
            Time_Second = int(event['gameTime'][-2:])

            if ("card" in event['label']): label = 1
            elif ("subs" in event['label']): label = 2
            elif ("soccer" in event['label']): label = 3
            else: label = 0

            if Half == testing_batch_features['half']+1:
                aroundValue = min(
                    Time_Minute * 60 + Time_Second,
                    Labels.shape[0] - 1
                )
                Labels[(aroundValue - int(sliding_window_seconds/2)):(aroundValue + int(sliding_window_seconds/2)), 0] = 0
                Labels[(aroundValue - int(sliding_window_seconds/2)):(aroundValue + int(sliding_window_seconds/2)), label] = 1

        testing_batch_labels = Labels

        self._current_testing_batch_index = num_batch

        return testing_batch_video_features, testing_batch_audio_features, testing_batch_labels, self.testing_GamesKeys[num_batch]
