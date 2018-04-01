import numpy as np
import librosa
import os

sumpath = 'speech_commands_v0.01'

commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

train_rate = 0.9

class KWSReader(object):

    def __init__(self, each_size=20, trainpath=sumpath, commands=commands):
        self.each_batch = each_size
        self.label2int = dict(zip(commands, range(len(commands))))
        self.label2int.update({'silence' : len(commands)})
        self.label2int.update({'unknown' : len(commands)+1})
        # read all wavs, if wav is in noise, concatenate them and make more data; else read each and append to 4000
        # append data via add noise
        self.raw_data = []
        self.environment_noise = None
        for i in range(len(os.listdir(sumpath))):
            self.raw_data.append([])
        count = 11
        for subw in os.listdir(sumpath):
            subfile = os.path.join(sumpath, subw)
            if subw == '_background_noise_':
                subw = 'silence'
                raww = None
                for ww in os.listdir(subfile):
                    wavpath = os.path.join(subfile, ww)
                    audio, fs = librosa.load(wavpath, sr=None)
                    if raww is None:
                        raww = audio
                    else:
                        raww = np.concatenate((raww, audio), axis=0)
                self.environment_noise = raww
                print(subw, ' read over')
            else:
                if subw in self.label2int:
                    idx = self.label2int[subw]
                else:
                    idx = count
                    count += 1
                subfile = os.path.join(sumpath, subw)
                for ww in os.listdir(subfile):
                    wavpath = os.path.join(subfile, ww)
                    audio, fs = librosa.load(wavpath, sr=None)
                    self.raw_data[idx].append(audio)
                print(subw, ' read over')

    def NoiseGet(self):
        start = np.random.randint(0, len(self.environment_noise)-16000)
        return self.environment_noise[start : start + 16000]*np.random.random()

    def Shift(self, audio, shift_ms=160):
        shift_point = 16000*shift_ms//1000
        if shift_point > len(audio)//2:
            shift_point = len(audio)//2
        shift_p = np.random.randint(-shift_point, shift_point)
        if shift_p > 0:
            return audio[shift_p:]
        else:
            return audio[:-shift_p]

    def OneAudioSample(self, audio):
        ret = []
        dice = np.random.random()
        if dice >= 0.2:
            audio = self.Shift(audio)
        #print(audio.shape)
        al = len(audio)
        if al < 16000:
            tobepad = 16000 - al
            front = np.random.randint(0, tobepad)
            a1 = np.pad(audio, (front, tobepad-front), 'constant')
        #print(a1.shape)
            ret.append(a1)
            en2 = self.NoiseGet()
            en2[front : front + al] = audio
        #print(en2.shape)
            ret.append(en2)
            en3 = self.NoiseGet()
            en3[front : front + al] += audio
        #print(en3.shape)
            ret.append(en3)
        else:
            ret.append(audio)
            en2 = audio + self.NoiseGet()
            ret.append(en2)
            en3 = audio + self.NoiseGet()
            ret.append(en3)
        return np.array(ret)

    def TrainBatch(self, unknownRate=0.2):
        labeld = self.each_batch * 11
        total = int(labeld/(1-unknownRate))
        ret_data = []
        ret_label = []
        for i in range(11):
            if i == self.label2int['silence']:
                for j in range(self.each_batch):
                    en = self.NoiseGet()
                    tmp = self.OneAudioSample(en)
                    #print(i, j, tmp.shape, en.shape)
                    ret_data.append(tmp)
                    ret_label.append(i)
            else:
                sltidx = np.random.choice(int(len(self.raw_data[i]) * train_rate), self.each_batch)
                for j in sltidx:
                    audio = self.raw_data[i][j]
                    tmp = self.OneAudioSample(audio)
                    #print(i, j, tmp.shape, audio.shape)
                    ret_data.append(tmp)
                    ret_label.append(i)
        for i in range(total - labeld):
            idx1 = np.random.randint(11, len(self.raw_data))
            idx2 = np.random.randint(0, int(len(self.raw_data[idx1]) * train_rate))
            audio = self.raw_data[idx1][idx2]
            tmp = self.OneAudioSample(audio)
            #print(idx1, idx2, tmp.shape, audio.shape)
            ret_data.append(tmp)
            ret_label.append(11)
        return np.array(ret_data), np.array(ret_label)

    def ValBatch(self, unknownRate=0.2):
        labeld = self.each_batch * 11
        total = int(labeld / (1 - unknownRate))
        ret_data = []
        ret_label = []
        for i in range(11):
            if i == self.label2int['silence']:
                for j in range(self.each_batch):
                    en = self.NoiseGet()
                    tmp = self.OneAudioSample(en)
                    # print(i, j, tmp.shape, en.shape)
                    ret_data.append(tmp)
                    ret_label.append(i)
            else:
                for j in range(self.each_batch):
                    idx = np.random.randint(int(len(self.raw_data[i]) * train_rate), len(self.raw_data[i]))
                    audio = self.raw_data[i][idx]
                    tmp = self.OneAudioSample(audio)
                    # print(i, j, tmp.shape, audio.shape)
                    ret_data.append(tmp)
                    ret_label.append(i)
        for i in range(total - labeld):
            idx1 = np.random.randint(11, len(self.raw_data))
            idx2 = np.random.randint(int(len(self.raw_data[idx1]) * train_rate), len(self.raw_data[idx1]))
            audio = self.raw_data[idx1][idx2]
            tmp = self.OneAudioSample(audio)
            # print(idx1, idx2, tmp.shape, audio.shape)
            ret_data.append(tmp)
            ret_label.append(11)
        return np.array(ret_data), np.array(ret_label)

