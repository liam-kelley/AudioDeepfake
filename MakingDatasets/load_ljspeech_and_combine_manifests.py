#https://pytorch.org/audio/stable/generated/torchaudio.datasets.LJSPEECH.html#torchaudio.datasets.LJSPEECH
#from torchaudio.datasets import LJSPEECH
from ljspeech_loader import LJSPEECH
import json
import string
import os

# Modified torchaudio.datasets.ljspeech.py to have encoding="utf8" in open()
dataset=LJSPEECH("LJSpeech-1.1")
#an ljspeech item is [audio_filepath, transcript, duration, normalized transcript]
#2713 ljspeech audio files == about 5 hours of audio
dicos_ljspeech = [{ "audio_filepath": str(dataset[i][0]).replace(os.sep, "/"), "text": dataset[i][1], "duration": dataset[i][2] } for i in range(2713)]

#print(dicos_ljspeech[0]["audio_filepath"]) #Check "/" replace is correct.

##with open("manifest_ljspeech_5h.json", "w") as outfile:
##    for dico in dicos_ljspeech:
##        json.dump(dico, outfile)
##        outfile.write("\n")
##print("wrote manifest_ljspeech_5h.json")

with open("manifest_train_cruise_only.json") as file:
    cruise_lines = [line.rstrip() + "\n" for line in file]
cruise_lines_len=len(cruise_lines) 

#combining cruise and ljspeech into the same manifest
with open("manifest_train.json", "w") as outfile:
    i=0
    for dico in dicos_ljspeech:
        #alternating one cruise sample for every ljspeech sample.
        #At some point we will run out of cruise files, so we will repeat them.
        outfile.write(cruise_lines[i%cruise_lines_len]) # cruise 
        json.dump(dico, outfile) # ljspeech 
        outfile.write("\n")
        i+=1
print("wrote manifest_train.json aka mainfest_train_COMBINED")


