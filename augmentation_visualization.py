import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import librosa
import noisereduce as nr

"""
A series of functions to help visualize how each individual audio file is augmented to produce
significantly more training data for the model to work with and test on.  Used for conceptually
understanding and manipulating the dataset, not a component of the overall model. Note: Includes 
lots of plots and not a lot of cleanliness or optimization
"""

def center(arr):
    mn = np.min(arr)
    return(arr-mn)

def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return((arr-mn) / (mx-mn))

def findNoise(audio, mn=0):
    # Split the audio clip over 20 even intervals
    step_size = int(audio.shape[0]/20)
    intervals = [[i, i + step_size] for i in range(0, audio.shape[0] - step_size, step_size)]
    
    # Find the variance of the change in amplitude over each interval and take the min as the noisy interval
    std = [np.std(center(audio[i[0]:i[1]])) for i in intervals]
    noisy_part = intervals[std.index(sorted(std)[mn])]
    noise = audio[noisy_part[0]:noisy_part[1]]
    return(noise)


def plot_noise_reduction(file):
    if file == '.DS_Store':
        return()
    audio,sr = librosa.load(file)
    step_size = int(audio.shape[0]/10)
    intervals = [[i, i + step_size] for i in range(0, audio.shape[0] - step_size, step_size)]
    std = [np.std(center(audio[i[0]:i[1]])) for i in intervals]
    noise_interval_1 = intervals[np.argmin(std)]
    noise_plot_1 = np.zeros(len(audio))
    noise_plot_1[noise_interval_1[0]:noise_interval_1[1]] = audio[noise_interval_1[0]:noise_interval_1[1]]
    
    # trim long audio clips, add silence to short audio clips    
    n = 616500
    if audio.shape[0] >= n:
        audio0,_  = librosa.effects.trim(audio,  top_db=20, frame_length=512, hop_length=64)
    else:
        audio0  = np.concatenate((audio,  np.zeros(n-audio.shape[0])))
    
    audio1 = nr.reduce_noise(audio0, findNoise(audio), verbose=False)
    audio2 = nr.reduce_noise(audio0, findNoise(audio)/1.5, verbose=False)  
    audio3 = nr.reduce_noise(audio0, findNoise(audio)/2, verbose=False)

    n_mels = 257
    mel_1 = librosa.feature.melspectrogram(audio0, sr=sr, n_fft=2048, hop_length=int(audio0.shape[0]/2000), n_mels=n_mels)
    mel_1 = librosa.power_to_db(mel_1, ref=np.max)
    mel_2 = librosa.feature.melspectrogram(audio1, sr=sr, n_fft=2048, hop_length=int(audio1.shape[0]/2000), n_mels=n_mels)
    mel_2 = librosa.power_to_db(mel_2, ref=np.max)
    mel_3 = librosa.feature.melspectrogram(audio2, sr=sr, n_fft=2048, hop_length=int(audio2.shape[0]/2000), n_mels=n_mels)
    mel_3 = librosa.power_to_db(mel_3, ref=np.max)
    mel_4 = librosa.feature.melspectrogram(audio3, sr=sr, n_fft=2048, hop_length=int(audio2.shape[0]/2000), n_mels=n_mels)
    mel_4 = librosa.power_to_db(mel_4, ref=np.max)
    
    fig = plt.figure(figsize=(12,8))
    fig.canvas.set_window_title('Augmenting Audio Dataset via Noise')
    
    ax1 = fig.add_subplot(4,2,1)
    ax1.tick_params(labelsize=6)
    ax1 = plt.plot(audio0, color='k', lw=0.5)
    plt.title('No Noise Reduction')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    ax7 = fig.add_subplot(4,2,3)
    ax7.tick_params(labelsize=6)
    ax7 = plt.plot(audio0, color='b', lw=0.5, alpha=0.3)
    ax7 = plt.plot(audio2, color='y', lw=0.5)
    ax7 = plt.plot(audio3, color='k', lw=0.5)
    ax7 = plt.plot(noise_plot_1, color='r', lw=0.5, alpha=0.9)
    plt.title('Noise Reduction 1')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    ax3 = fig.add_subplot(4,2,5)
    ax3.tick_params(labelsize=6)
    ax3 = plt.plot(audio0, color='b', lw=0.5, alpha=0.3)
    ax3 = plt.plot(audio1, color='y', lw=0.5)
    ax3 = plt.plot(audio2, color='k', lw=0.5)
    ax3 = plt.plot(noise_plot_1, color='r', lw=0.5, alpha=0.9)
    plt.title('Noise Reduction 2')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    ax2 = fig.add_subplot(4,2,7)
    ax2.tick_params(labelsize=6)
    ax3 = plt.plot(audio0, color='b', lw=0.5, alpha=0.3)
    ax2 = plt.plot(audio1, color='k', lw=0.5)
    ax2 = plt.plot(noise_plot_1, color='r', lw=0.5, alpha=0.9)
    plt.title('Noise Reduction 3')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    ax4 = fig.add_subplot(4,2,2)
    ax4.tick_params(labelsize=6)
    ax4 = sns.heatmap(mel_1, cbar=False, cmap='gray')
    ax4.invert_yaxis()
    plt.title('Audio Data Augmentation via Noise', weight='bold')
    plt.xlabel('Time')
    plt.ylabel('Mels')
    
    ax8 = fig.add_subplot(4,2,4)
    ax8.tick_params(labelsize=6)
    ax8 = sns.heatmap(mel_4, cbar=False, cmap='gray') 
    ax8.invert_yaxis()
    plt.xlabel('Time')
    plt.ylabel('Mels')

    ax6 = fig.add_subplot(4,2,6)
    ax6.tick_params(labelsize=6)
    ax6 = sns.heatmap(mel_3, cbar=False, cmap='gray') 
    ax6.invert_yaxis()
    plt.xlabel('Time')
    plt.ylabel('Mels')
    
    ax5 = fig.add_subplot(4,2,8)
    ax5.tick_params(labelsize=6)
    ax5 = sns.heatmap(mel_2, cbar=False, cmap='gray') 
    ax5.invert_yaxis()
    plt.xlabel(label)
    plt.ylabel('Mels')

    plt.tight_layout()
    
    
def plot_translations(file):
    if file == '.DS_Store':
        return()
    audio,sr = librosa.load(file)
    audio = nr.reduce_noise(audio, findNoise(audio), verbose=False)

    # trim long audio clips, add silence to short audio clips    
    n = 616500
    if audio.shape[0] >= n:
        audio0,_  = [librosa.effects.trim(audio,  top_db=20, frame_length=512, hop_length=64)]
    else:
        padding = n - audio.shape[0]
        m = 12 # divide the space into m partitions
        partition_const = m/n  # this num determines how many translations we should have out of m max based on the length of the file
        num_partitions = int(m - round(audio.shape[0]*partition_const, 0))
        partition =  np.linspace(0,padding,num_partitions)
        audio0 = []
        for i in range(num_partitions):
            # translate the spectrogram over the partitions with a bit of randomness
            k = partition[i]
            rand = random.uniform(-0.4,0.4)
            if i != 0 and i != num_partitions-1:
                shift = k + partition[1]*rand
            else:
                shift = k
            audio0.append(np.concatenate((np.zeros(int(shift)), audio,  np.zeros(padding-int(shift)))))
                          
    n_mels = 257
    mel_1 = librosa.feature.melspectrogram(audio0[0], sr=sr, n_fft=2048, hop_length=int(audio0[0].shape[0]/2000), n_mels=n_mels)
    mel_1 = librosa.power_to_db(mel_1, ref=np.max)
    mel_2 = librosa.feature.melspectrogram(audio0[1], sr=sr, n_fft=2048, hop_length=int(audio0[1].shape[0]/2000), n_mels=n_mels)
    mel_2 = librosa.power_to_db(mel_2, ref=np.max)
    mel_3 = librosa.feature.melspectrogram(audio0[2], sr=sr, n_fft=2048, hop_length=int(audio0[2].shape[0]/2000), n_mels=n_mels)
    mel_3 = librosa.power_to_db(mel_3, ref=np.max)
    mel_4 = librosa.feature.melspectrogram(audio0[3], sr=sr, n_fft=2048, hop_length=int(audio0[3].shape[0]/2000), n_mels=n_mels)
    mel_4 = librosa.power_to_db(mel_4, ref=np.max)
    mel_5 = librosa.feature.melspectrogram(audio0[4], sr=sr, n_fft=2048, hop_length=int(audio0[4].shape[0]/2000), n_mels=n_mels)
    mel_5 = librosa.power_to_db(mel_5, ref=np.max)

    
    fig = plt.figure(figsize=(10,8))
    fig.canvas.set_window_title('Augmenting Audio Dataset via Noise')
    
    ax1 = fig.add_subplot(5,1,1)
    ax1.tick_params(labelsize=6)
    ax1 = sns.heatmap(mel_1, cbar=False)
    ax1.invert_yaxis()
    plt.title('Audio Data Augmentation via Psuedo Random Time Translation', weight='bold')
    plt.xlabel('Time')
    plt.ylabel('Mels')
    
    ax2 = fig.add_subplot(5,1,2)
    ax2.tick_params(labelsize=6)
    ax2 = sns.heatmap(mel_2, cbar=False) 
    ax2.invert_yaxis()
    plt.xlabel('Time')
    plt.ylabel('Mels')
    
    ax3 = fig.add_subplot(5,1,3)
    ax3.tick_params(labelsize=6)
    ax3 = sns.heatmap(mel_3, cbar=False) 
    ax3.invert_yaxis()
    plt.xlabel('Time')
    plt.ylabel('Mels')
    
    ax4 = fig.add_subplot(5,1,4)
    ax4.tick_params(labelsize=6)
    ax4 = sns.heatmap(mel_4, cbar=False) 
    ax4.invert_yaxis()
    plt.xlabel('Time')
    plt.ylabel('Mels')

    ax5 = fig.add_subplot(5,1,5)
    ax5.tick_params(labelsize=6)
    ax5 = sns.heatmap(mel_5, cbar=False) 
    ax5.invert_yaxis()
    plt.xlabel('Sound Class: ' + label.strip('/'))
    plt.ylabel('Mels')
    plt.tight_layout()

    
""" SAMPLE RANDOM AUDIO FILES """
directory = r'UrbanSound/data/'
folders = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']

# Plot the noise reduction of three random audio clips from the directory
num_samples = 1
k = 0
while k < num_samples:
    label = random.choice(folders) + '/'
    label = 'toilet/'
    index = random.randint(0, len(os.listdir(directory + label))-1)
    file = os.listdir(directory + label)[index]
    
    path = directory + label + file
    plot_noise_reduction(path)
    plot_translations(path)
    
    #plot_MEL_spectrogram('UrbanSound/data/air_conditioner/63724.wav')    
    k+=1
    
