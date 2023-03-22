# AudioDeepfake
 Finetuning a FastPitch and HifiGan model on a Tom cruise Dataset I made from [this interview](https://www.youtube.com/watch?v=P_1TZ4gYA2s) .

### Getting started.

1. Unzip files cruise_audios_22k.zip and LJSpeech-1.1_small.zip
2. install depedencies for cruisetuning (use a venv or something) 
3. python fastpitch_cruisetuning.py
4. python infer_some_cruise_audio.py
5. read your audio sample yeahh

The following is to finetune the vocoder, and isn't done.

6. python synthetize_mels_from_fastpitch_for_hifigan_cruisetuning.py (probably need to debug that code)
7. python hifigan_cruisetuning.py
8. modify infer_some_cruise_audio.py to load hifigan from checkpoint.
