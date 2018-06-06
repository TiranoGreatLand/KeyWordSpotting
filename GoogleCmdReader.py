import numpy as np
import os
import librosa
from sklearn.cross_validation import KFold

cmd_path = 'KWS/speech_commands_v0.01'

test_cmds = ['tree', 'three', 'wow']
test_cmds = set(test_cmds)

max_audio_length = 16000

standard_max_shape = librosa.feature.mfcc(np.arange(max_audio_length), sr=16000, n_mfcc=64).T.shape
standard_max_timestep = standard_max_shape[0]
print(standard_max_timestep)
max_seq_number = 6

def PreProcess(audio):
    mfccs = librosa.feature.mfcc(audio, sr=16000, n_mfcc=64).T
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    seqlen = mfccs.shape[0]
    return mfccs, seqlen

class DataBase(object):

    def __init__(self, cmd_path=cmd_path, batch_size=8):
        self.cmd_path = cmd_path
        self.batch_size = batch_size

    def DataStatistic(self):
        lengthes  = []
        nums = []
        for word_name in os.listdir(self.cmd_path):
            words_waves_folder = os.path.join(self.cmd_path, word_name)
            nums.append(len(os.listdir(words_waves_folder)))
            for wave_file in os.listdir(words_waves_folder):
                wave_path = os.path.join(words_waves_folder, wave_file)
                audio, sr = librosa.load(wave_path, sr=16000)
                al = len(audio)
                lengthes.append(al)
            print(word_name, ' read over')
        print("statistic result ")
        print("mean ", np.mean(lengthes)/16000)
        print("std  ", np.std(lengthes)/16000)
        print("max  ", np.max(lengthes)/16000)
        print("min  ", np.min(lengthes)/16000)
        print("numbers ")
        print("mean ", np.mean(nums))
        print("std  ", np.std(nums))
        print("max  ", np.max(nums))
        print("min  ", np.min(nums))
        print("totally ", np.sum(nums))

    def ReadAndTransfer(self):
        char_number_dict = {}

        mfcc_data = []
        seqlens = []
        train_words = []
        train_labels = []

        test_md = []
        test_seqlens = []
        test_words = []
        test_labels = []

        train_count = 0
        test_count = 0
        for word_name in os.listdir(self.cmd_path):
            words_waves_folder = os.path.join(self.cmd_path, word_name)
            next_file = os.listdir(words_waves_folder)[:100]
            num_next_file = len(next_file)
            print(word_name, num_next_file)

            if word_name != 'tree' and word_name != 'three' and word_name != 'wow':
                the_label = train_count
                train_count += 1
                for c in word_name:
                    if c not in char_number_dict:
                        char_number_dict[c] = num_next_file
                    else:
                        char_number_dict[c] += num_next_file
            else:
                the_label = test_count
                test_count += 1

            tmp_mfcc_data = np.zeros((num_next_file, standard_max_timestep, 64), np.float32)
            tmp_seqlens = []
            tmp_label = []
            for i, wave_file in enumerate(next_file):
                wave_path = os.path.join(words_waves_folder, wave_file)
                audio, sr = librosa.load(wave_path, sr=16000)
                md, sl = PreProcess(audio)
                tmp_mfcc_data[i, :sl, :] = md
                tmp_seqlens.append(sl)
                tmp_label.append(the_label)
            if word_name == 'tree' or word_name == 'three' or word_name == 'wow':
                test_words.append(word_name)
                test_md.append(tmp_mfcc_data)
                test_seqlens.append(np.asarray(tmp_seqlens))
                test_labels.append(tmp_label)
            else:
                train_words.append(word_name)
                mfcc_data.append(tmp_mfcc_data)
                seqlens.append(tmp_seqlens)
                train_labels.append(tmp_label)

        count_pairs = sorted(char_number_dict.items(), key=lambda x: x[1])
        print(len(count_pairs))
        for n in count_pairs:
            print(n)
        words, _ = zip(*count_pairs)
        self.word_num_map = dict(zip(words, range(len(words))))
        self.num_word_map = {}
        for w in self.word_num_map:
            num = self.word_num_map[w]
            self.num_word_map[num] = w
        # data manipulation
        label_vectors = []
        for i, ws in enumerate(train_words):
            num_seq = self.CharSeq2NumSeq(ws)
            tmp_seqs = []
            tmpl = len(mfcc_data[i])
            for x in range(tmpl):
                tmp_seqs.append(num_seq)
            label_vectors.append(tmp_seqs)

        self.train_data = np.concatenate(mfcc_data, axis=0)
        self.train_seqlens = np.concatenate(seqlens, axis=0)
        self.train_label_vectors = np.concatenate(label_vectors, axis=0)
        self.train_label = np.concatenate(train_labels, axis=0)

        test_lblvs = []
        for i, ws in enumerate(test_words):
            num_seq = self.CharSeq2NumSeq(ws)
            tmp_seqs = []
            tmpl = len(test_md[i])
            for x in range(tmpl):
                tmp_seqs.append(num_seq)
            test_lblvs.append(tmp_seqs)

        self.test_data = np.concatenate(test_md, axis=0)
        self.test_seqlens = np.concatenate(test_seqlens, axis=0)
        self.test_label_vectors = np.concatenate(test_lblvs, axis=0)
        self.test_label = np.concatenate(test_labels, axis=0)

        kfold_10 = KFold(len(self.train_data), 10, True)
        slt_n = np.random.randint(10)
        for i, (tridxs, validxs) in enumerate(kfold_10):
            if i == slt_n:
                self.val_data = self.train_data[validxs]
                self.val_seqlens = self.train_seqlens[validxs]
                self.val_label_vectors = self.train_label_vectors[validxs]
                self.val_label = self.train_label[validxs]
                self.train_data = self.train_data[tridxs]
                self.train_seqlens = self.train_seqlens[tridxs]
                self.train_label_vectors = self.train_label_vectors[tridxs]
                self.train_label = self.train_label[tridxs]
                break
        print("data manipulate over")
        print("train data : ", self.train_data.shape, self.train_seqlens.shape, self.train_label_vectors.shape)
        print("val data : ", self.val_data.shape, self.val_seqlens.shape, self.val_label_vectors.shape)
        print("test data : ", self.test_data.shape, self.test_seqlens.shape, self.test_label_vectors.shape)
        self.len_train_data = len(self.train_data)
        self.len_val_data = len(self.val_data)
        self.len_test_data = len(self.test_data)

    def CharSeq2NumSeq(self, charseq):
        numberseq = []
        for c in charseq:
            n = self.word_num_map[c]
            numberseq.append(n)
        if len(numberseq) < max_seq_number:
            tmp = -np.ones(max_seq_number)
            tmp[ : len(numberseq)] = numberseq
            return tmp
        else:
            return np.asarray(numberseq)

    def NumSeq2CharSeq(self, numseq):
        string = ''
        for n in numseq:
            if n == -1:
                break
            elif n in self.num_word_map:
                string += self.num_word_map[n]
        return string

#db = DataBase()
#db.DataStatistic()