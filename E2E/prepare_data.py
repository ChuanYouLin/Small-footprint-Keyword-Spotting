# from python_speech_features import mfcc
from torch.utils.data import Dataset, DataLoader
import librosa
import random
import sys
import os
import re
import hashlib
import numpy as np
import torch

keyword = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

    Returns:
    String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # hash_name = re.sub(r'.wav', '', hash_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode("utf-8")).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
            (MAX_NUM_WAVS_PER_CLASS + 1)) *
            (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < float(validation_percentage):
        result = 'validation'
    elif percentage_hash < float(testing_percentage) + float(validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

def load_one_data(path):
    y, sr = librosa.load(path, sr = 16000)
    S = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 512, hop_length = 128)
    mfcc_li = librosa.feature.mfcc(S = librosa.power_to_db(S), sr = sr, n_mfcc = 40)
    mfcc_feat = mfcc_li.transpose()
    
    length = mfcc_feat.shape[0]
    # mfcc -> 99*40 , mfcc_li -> 126*40
    padded_audio = np.zeros([126, 40])
    padded_audio[:length] = mfcc_feat
    return padded_audio

def load_data(mypath, output_path):
    validation_percentage, testing_percentage = 10, 10
    files =  sorted(os.listdir(mypath))
    files.remove('_background_noise_')
    fullpath_files = []
    for f in files:
        fullpath = os.path.join(mypath, f)
        if os.path.isdir(fullpath):
            fullpath_files.append(fullpath)
    data = {}
    spk_dict = {}
    data['training'] = []
    data['validation'] = []
    data['testing'] = []
    for f in fullpath_files:
        audios = sorted(os.listdir(f))
        for a in audios:
            fullpath = os.path.join(f, a)
            data_active = which_set(fullpath, validation_percentage, testing_percentage)
            data[data_active].append(fullpath)
            
            spk = a.split('_')[0]
            if spk not in spk_dict:
                spk_dict[spk] = 1
            else:
                spk_dict[spk] += 1
    
    spk_dict_ten = spk_dict.copy()
    spk_dict_ten['else'] = 0
    for key in spk_dict:
        if spk_dict[key] < 10:
            spk_dict_ten['else'] += spk_dict_ten[key]
            del spk_dict_ten[key]

    training_X = []
    training_Y_SPK = []
    training_Y_KEYWORD = []
    for fullpath in data['training']:
        text = fullpath.split('/')[-2]
        y, sr = librosa.load(fullpath, sr = 16000)
        S = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 512, hop_length = 128)
        mfcc_li = librosa.feature.mfcc(S = librosa.power_to_db(S), sr = sr, n_mfcc = 40)
        mfcc_feat = mfcc_li.transpose()
        
        length = mfcc_feat.shape[0]
        # mfcc -> 99*40 , mfcc_li -> 126*40
        padded_audio = np.zeros([126, 40])
        padded_audio[:length] = mfcc_feat
        if text in keyword:
            label_keyword = keyword.index(text)
        else:
            label_keyword = len(keyword)

        spk = fullpath.split('/')[-1].split('_')[0]
        if spk not in spk_dict_ten:
            spk = 'else'
        label_spk = list(spk_dict_ten.keys()).index(spk)

        training_X.append(padded_audio)
        training_Y_SPK.append(label_spk)
        training_Y_KEYWORD.append(label_keyword)

    np.save(os.path.join(output_path, "train_X.npy"), training_X)
    np.save(os.path.join(output_path, "train_Y_spk.npy"), training_Y_SPK)
    np.save(os.path.join(output_path, "train_Y_text.npy"), training_Y_KEYWORD)

    validation_X = []
    validation_Y_SPK = []
    validation_Y_KEYWORD = []
    for fullpath in data['validation']:
        text = fullpath.split('/')[-2]
        y, sr = librosa.load(fullpath, sr = 16000)
        S = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 512, hop_length = 128)
        mfcc_li = librosa.feature.mfcc(S = librosa.power_to_db(S), sr = sr, n_mfcc = 40)
        mfcc_feat = mfcc_li.transpose()
        
        length = mfcc_feat.shape[0]
        # mfcc -> 99*40 , mfcc_li -> 126*40
        padded_audio = np.zeros([126, 40])
        padded_audio[:length] = mfcc_feat
        if text in keyword:
            label_keyword = keyword.index(text)
        else:
            label_keyword = len(keyword)

        spk = fullpath.split('/')[-1].split('_')[0]
        if spk not in spk_dict_ten:
            spk = 'else'
        label_spk = list(spk_dict_ten.keys()).index(spk)

        validation_X.append(padded_audio)
        validation_Y_SPK.append(label_spk)
        validation_Y_KEYWORD.append(label_keyword)

    np.save(os.path.join(output_path, "validation_X.npy"), validation_X)
    np.save(os.path.join(output_path, "validation_Y_spk.npy"), validation_Y_SPK)
    np.save(os.path.join(output_path, "validation_Y_text.npy"), validation_Y_KEYWORD)

    test_X = []
    test_Y_SPK = []
    test_Y_KEYWORD = []
    for fullpath in data['testing']:
        text = fullpath.split('/')[-2]
        y, sr = librosa.load(fullpath, sr = 16000)
        
        S = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 512, hop_length = 128)
        mfcc_li = librosa.feature.mfcc(S = librosa.power_to_db(S), sr = sr, n_mfcc = 40)
        mfcc_feat = mfcc_li.transpose()
        
        length = mfcc_feat.shape[0]
        # mfcc -> 99*40 , mfcc_li -> 126*40
        padded_audio = np.zeros([126, 40])
        padded_audio[:length] = mfcc_feat
        if text in keyword:
            label_keyword = keyword.index(text)
        else:
            label_keyword = len(keyword)

        spk = fullpath.split('/')[-1].split('_')[0]
        if spk not in spk_dict_ten:
            spk = 'else'
        label_spk = list(spk_dict_ten.keys()).index(spk)

        test_X.append(padded_audio)
        test_Y_SPK.append(label_spk)
        test_Y_KEYWORD.append(label_keyword)

    np.save(os.path.join(output_path, "test_X.npy"), test_X)
    np.save(os.path.join(output_path, "test_Y_spk.npy"), test_Y_SPK)
    np.save(os.path.join(output_path, "test_Y_text.npy"), test_Y_KEYWORD)

def main():
    load_data(sys.argv[1], sys.argv[2])


if __name__ == "__main__": main()