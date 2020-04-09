import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import noisereduce as nr
import librosa
import random
from PIL import Image  
import PIL  

"""
This algorithm augments the data set by creating 3 copies of each audio file with varying levels of noise and generates
several additional copies where the audio files are psuedo-randomly translated accross their time axes.  

Then features are extracted using mel spectrograms, or MFCC.  Each MFCC is then saved as a jpg and stored in a new
directory to be used by the CNN
"""

# Utils
def center(arr):
    mn = np.min(arr)
    return(arr-mn)

def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return((arr-mn) / (mx-mn))

def findNoise(audio):
    # Split the audio clip into 8 even intervals
    step_size = int(audio.shape[0]/12)
    intervals = [[i, i + step_size] for i in range(0, audio.shape[0] - step_size, step_size)]
    
    # Find the std of the centered amplitudes over each interval and take the interval with smallest std as the noise
    std = [np.std(center(audio[i[0]:i[1]])) for i in intervals]
    noisy_part = intervals[np.argmin(std)]
    noise = audio[noisy_part[0]:noisy_part[1]]
    return(noise)

def save_spectrogram(arr, path):
    fig = plt.figure(figsize=(5,0.6425)) # shape the output to be exactly 2000x256
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax = fig.add_subplot(1,1,1)
    ax.axis('off')
    spec = sns.heatmap(arr, cmap='gray', cbar=False)
    ax.invert_yaxis()
    ax.axis('off')
    fig.savefig(path, dpi=400, frameon='false', pil_kwargs={'L'})
    plt.close(fig)
    Image.open(path).convert('L').save(path)
    return()

def translate(audio, n = 616500):
    # Augment by translating if clip is short enough
    padding = n - audio.shape[0]
    m = 12 # divide the space into m partitions
    partition_const = m/n  # this num determines how many translations we should have out of m max based on the length of the file
    num_partitions = m - int(audio.shape[0]*partition_const)
    partition =  np.linspace(0,padding,num_partitions)
    translations = []
    
    # translate the spectrogram over the partition with a bit of randomness
    for i in range(num_partitions):
        k = partition[i]
        rand = random.uniform(-0.4,0.4)
        if i != 0 and i != num_partitions-1:
            shift = k + partition[1]*rand
        else:
            shift = k
        translations.append(np.concatenate((np.zeros(int(shift)), audio,  np.zeros(padding-int(shift)))))
    return(translations)

def processAudio(file, path):
    audio,sr = librosa.load(file) # raw audio file

    # trim long audio clips, add silence to short audio clips    
    n = 616500 # correspondw to approx 30 seconds, 1 sec ~ 20550 n
    if audio.shape[0] >= 4*n:  # ignore all files > 2 mins
        print(f'Skipped, clip was {audio.shape[0]/(2*n)} mins long')
        return()
    elif audio.shape[0] >= n:
        audio,_ = librosa.effects.trim(audio, top_db=20, frame_length=512, hop_length=64)
        audio = audio[:n-1]
    
    # Augment the audio with varying levels of noise
    audio1 = nr.reduce_noise(audio, findNoise(audio), verbose=False) # de-noised most
    audio2 = nr.reduce_noise(audio, findNoise(audio)/1.5, verbose=False) # de-noised less
    audio3 = nr.reduce_noise(audio, findNoise(audio)/2, verbose=False) # de-noised least
    # Augment the audio by translating each sample wrt the time axis
    audio = translate(audio1) + translate(audio2) + translate(audio3)
    
    # Convert each audio file into a Mel Spectrogram and save it
    n_mels = 257
    k = 0
    path = path.replace('.' + path.split('.')[-1],'')
    for clip in audio:
        mel = librosa.feature.melspectrogram(clip, sr=sr, n_fft=2048, hop_length=int(clip.shape[0]/2000), n_mels=n_mels)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = minMaxNormalize(mel)
        mel = mel[0:258,0:2000]  #shape is 257 x 2000
        image_path = path + f'-{k}.jpg'
        print(k)
        save_spectrogram(mel, image_path) # save a black & white version of the spectrogram
        k+=1
    return()


""" MAIN """
directory = r'UrbanSound/data/'
image_dir = r'training_images/'
labels = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']
master_df = pd.DataFrame()

for folder in os.listdir(directory):
    if folder == '.DS_Store':
        continue
    k = 0
    #folder = 'toilet'
    for file in os.listdir(directory + '/' + folder + '/'):
        if file == '.DS_Store' or (folder == 'siren' and k<57):
            continue
        path = directory + '/' + folder + '/' + file   
        image_path = image_dir + folder + '/' + file
        processAudio(path, image_path)
        k+=1
