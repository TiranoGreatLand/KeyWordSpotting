import numpy as np
import librosa
import os

sumpath = 'speech_commands_v0.01'

all_length = []
sub_length = {}

for subw in os.listdir(sumpath):
    count = 0
    if subw not in sub_length:
        sub_length[subw] = []
    nextp = os.path.join(sumpath, subw)
    if subw == '_background_noise_':
        for ww in os.listdir(nextp):
            wavpath = os.path.join(nextp, ww)
            audio, fs = librosa.load(wavpath, sr=None)
            al = len(audio)
            sub_length[subw].append(al)
            if count == 0:
                print(np.max(audio), np.min(audio), np.mean(audio), np.std(audio))
                count += 1
    else:
        for ww in os.listdir(nextp):
            wavpath = os.path.join(nextp, ww)
            audio, fs = librosa.load(wavpath, sr=None)
            al = len(audio)
            sub_length[subw].append(al)
            all_length.append(al)
            if count == 0:
                print(np.max(audio), np.min(audio), np.mean(audio), np.std(audio))
                count += 1
    print(subw, ' read over ')

print("all the data except background noise")
print(np.mean(all_length)/16000, np.std(all_length)/16000, np.max(all_length)/16000, np.min(all_length)/16000)

print("each commands")
for w in sub_length:
    print(w, len(sub_length[w]), np.mean(sub_length[w])/16000, np.std(sub_length[w])/16000, np.max(sub_length[w])/16000, np.min(sub_length[w])/16000)