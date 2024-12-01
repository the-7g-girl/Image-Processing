import cv2
import numpy as np
import librosa

def solution(audio_path):
    ############################
    ############################
    y, sr = librosa.load(audio_path, sr=None)
    discrete_ft = np.fft.fft(y)
    n = len(y)
    discrete_freq = np.fft.fftfreq(n, 1/sr)
    max_amplitude_index = np.argmax(np.abs(discrete_ft))
    max_amplitude_freq = discrete_freq[max_amplitude_index]

    if max_amplitude_freq > 350:
        class_name = 'metal'
    else:
        class_name = 'cardboard'
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # class_name = 'cardboard'
    return class_name
