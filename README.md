# AudioDeepfake
Finetuning a FastPitch and HifiGan model on a Tom Cruise Dataset I made from [this interview](https://www.youtube.com/watch?v=P_1TZ4gYA2s).
This repo reworks [NVidia's fastpitch finetuning tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/FastPitch_Finetuning.ipynb).

### Getting started with this repo

0. Clone repo
1. Unzip files cruise_audios_22k.zip and LJSpeech-1.1_small.zip
2. install depedencies for cruisetuning (use a venv or something) 
3. python fastpitch_cruisetuning.py
4. python infer_some_cruise_audio.py
5. read your audio sample yeahh

The following is to finetune the vocoder, and isn't tested yet.

6. python synthetize_mels_from_fastpitch_for_hifigan_cruisetuning.py (probably need to debug that code)
7. python hifigan_cruisetuning.py
8. modify infer_some_cruise_audio.py to load hifigan from checkpoint.

### The dataset

The Dataset has 376 Tom Cruise Samples. Total length is a little over 20 minutes. The data is transcribed in the manifests folder. Here is info on how I made the dataset:

##### The CruiseSet

0. Audio downloaded using a cheap youtube-mp3 downloader.
1. Preprocessing on FL Studio : Light denoising using Edison, Low cut at 56Hz, Compression using FabFilter Pro-C2 {mode: clean, Threshold : -15.44dB, Ratio : 75,3 to 1, Attack : 0.25ms, Release : 209,2 ms, Knee: +9.48dB, Range : +60dB, Hold: 0.00ms}, Band Compression using FabFilter Pro-MB { 1 band, centered at 57.765Hz, High Crossover : 111.23Hz, Mode: Expand, Threshold : -36.44dB, Ratio : 4.00 to 1, Attack : 10.6%, Release : 30.2%, Knee: 24dB, Range : -5.84dB}
2. Manually cut cruise samples to last between 1 and 6 seconds.
3. Export audios as mono 16bit wavs.
4. Resampled audio as 22050Hz from 44100Hz using MakingDatsets/resample_audio.py
5. Transcribed all sample's texts using OpenAI's Whisper Api on Colab (MakingDatasets/CruiseSetWhisper.ipnyb), exported and downloaded a first version of a cruise manifest.json file (with durations!) that way.
6. Additionnal manifest.json editing using good ol' notepad cntrl-H search & replace (specifically chose cool lines for validation)

##### Combining the CruiseSet with LJSpeech for FastPitch Training

1. Modified Torchaudio's ljspeech.py file to better fit this project's manifest creating needs --> ljspeech_loader.py
2. Selected about 5h of audio from LJSpeech.
3. Created a combined+alternating manifest_train_mult_speakers.json file using load_ljspeech_and_combine_manifests.json.
