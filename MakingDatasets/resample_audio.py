import numpy as np
import glob
import librosa
import soundfile as sf

# print(sorted(glob.glob("./cruise_audios/cruise*")))

i=1
for audiofile in sorted(glob.glob("./cruise_audios/cruise*")):
    y, sr = librosa.load(audiofile, sr=44100)
    y_22k = librosa.resample(y, orig_sr=sr, target_sr=22050)
    # Write out audio as 24bit PCM WAV
    sf.write('./cruise_audios_22k/cruise'+ f"{i:03}" +'.wav', y_22k, 22050, subtype='PCM_16')
    i+=1
    print(i,"nice")



