# -*- coding: utf-8 -*-
import os

os.system('python fastpitch_finetune.py --config-name=fastpitch_cruisetuning_v2.yaml \
  model.train_ds.dataloader_params.batch_size=24 model.validation_ds.dataloader_params.batch_size=24 \
  model.pitch_mean=100.3 model.pitch_std=64.0 \
  model.pitch_fmin=30 model.pitch_fmax=512 model.optim.lr=2e-4 \
  ~model.optim.sched model.optim.name=adam trainer.devices=1 trainer.strategy=null \
  +model.text_tokenizer.add_blank_at=true \
)')

"""## Improving Speech Quality

We see that from fine-tuning FastPitch, we were able to generate audio in a male voice but the audio quality is not as good as we expect. We recommend two steps to improve audio quality:

* Finetuning HiFi-GAN
* Adding more data

**Note that both of these steps are outside the scope of the notebook due to the limited compute available on Colab, but the code is included below for you to use outside of this notebook.**

### Finetuning HiFi-GAN
From the synthesized samples, there might be audible audio crackling. To fix this, we need to finetune HiFi-GAN on the new speaker's data. HiFi-GAN shows improvement using **synthesized mel spectrograms**, so the first step is to generate mel spectrograms with our finetuned FastPitch model to use as input.

The code below uses our finetuned model to generate synthesized mels for the training set we have been using. You will also need to do the same for the validation set (code should be very similar, just with paths changed).
"""








"""We can then finetune hifigan similarly to fastpitch using NeMo's [hifigan_finetune.py](https://github.com/NVIDIA/NeMo/blob/main/examples/tts/hifigan_finetune.py) and [hifigan.yaml](https://github.com/NVIDIA/NeMo/blob/main/examples/tts/conf/hifigan/hifigan.yaml):

```bash
python examples/tts/hifigan_finetune.py \
--config-name=hifigan.yaml \
model.train_ds.dataloader_params.batch_size=32 \
model.max_steps=1000 \
model.optim.lr=0.00001 \
~model.optim.sched \
train_dataset=./hifigan_train_ft.json \
validation_datasets=./hifigan_val_ft.json \
exp_manager.exp_dir=hifigan_ft \
+init_from_pretrained_model=tts_hifigan \
trainer.check_val_every_n_epoch=10 \
model/train_ds=train_ds_finetune \
model/validation_ds=val_ds_finetune
```

Like when finetuning FastPitch, we lower the learning rate and get rid of the optimizer schedule for finetuning. You will need to create `<your_hifigan_val_manifest>` and the synthesized mels corresponding to it.

As mentioned, the above command is not runnable in Colab due to limited compute resources, but you are free to finetune HiFi-GAN on your own machines.

### Adding more data
We can add more data in two ways. They can be combined for the best effect:

* **Add more training data from the new speaker**

The entire notebook can be repeated from the top after a new JSON manifest is defined that includes the additional data. Modify your finetuning commands to point to the new manifest. Be sure to increase the number of steps as more data is added to both the FastPitch and HiFi-GAN finetuning.

We recommend **1000 steps per minute of audio for fastpitch and 500 steps per minute of audio for HiFi-GAN**.

* **Mix new speaker data with old speaker data**

We recommend finetuning FastPitch (but not HiFi-GAN) using both old speaker data (LJSpeech in this notebook) and the new speaker data. In this case, please modify the JSON manifests when finetuning FastPitch to include speaker information by adding a `speaker` field to each entry:

```
{"audio_filepath": "new_speaker.wav", "text": "sample", "duration": 2.6, "speaker": 1}
{"audio_filepath": "old_speaker.wav", "text": "LJSpeech sample", "duration": 2.6, "speaker": 0}
```

5 hours of data from the old speaker should be sufficient for training; it's up to you how much data from the old speaker to use in validation.

For the training manifest, since we likely have less data from the new speaker, we need to ensure that the model sees a similar amount of new data and old data. We can do this by repeating samples from the new speaker until we have an equivalent number of samples from the old and new speaker. For each sample from the old speaker, please add a sample from the new speaker in the .json.

As a toy example, if we use 4 samples of the old speaker and only 2 samples of the new speaker, we would want the order of samples in our manifest to look something like this:

```
old_speaker_sample_0
new_speaker_sample_0
old_speaker_sample_1
new_speaker_sample_1
old_speaker_sample_2
new_speaker_sample_0  # Start repeat of new speaker samples
old_speaker_sample_3
new_speaker_sample_1
```

Once the manifests are created, we can modify the FastPitch training command to point to the new training and validation JSON files.

We also need to update `model.n_speakers=1` to `model.n_speakers=2`, as well as update the `sup_data_types` specified in the config file to include `speaker_id` (`sup_data_types=[align_prior_matrix,pitch,speaker_id]`). Updating these two fields is very important--otherwise the model will not recognize that there is more than one speaker!

Ensure the pitch statistics correspond to the new speaker rather than the old speaker for best results.

**For HiFiGAN finetuning, the training should be done on the new speaker data.**
"""
