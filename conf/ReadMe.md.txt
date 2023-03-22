###Let's take a closer look at the training command:

* `--config-name=fastpitch_align_v1.05.yaml`
  * We first tell the script what config file to use.

  sup_data_path=./fastpitch_sup_data`
  * We tell the script what manifest files to train and eval on, as well as where supplementary data is located (or will be calculated and saved during training if not provided).
  
* `phoneme_dict_path=tts_dataset_files/cmudict-0.7b_nv22.10 
heteronyms_path=tts_dataset_files/heteronyms-052722
whitelist_path=tts_dataset_files/tts.tsv 
`
  * We tell the script where `phoneme_dict_path`, `heteronyms-052722` and `whitelist_path` are located. These are the additional files we downloaded earlier, and are used in preprocessing the data.
  
* `exp_manager.exp_dir=./ljspeech_to_9017_no_mixing_5_mins`
  * Where we want to save our log files, tensorboard file, checkpoints, and more.

* `+init_from_nemo_model=./tts_en_fastpitch_align.nemo`
  * We tell the script what checkpoint to finetune from.

* `+trainer.max_steps=1000 ~trainer.max_epochs trainer.check_val_every_n_epoch=25`
  * For this experiment, we tell the script to train for 1000 training steps/iterations rather than specifying a number of epochs to run. Since the config file specifies `max_epochs` instead, we need to remove that using `~trainer.max_epochs`.

* `model.train_ds.dataloader_params.batch_size=24 model.validation_ds.dataloader_params.batch_size=24`
  * Set batch sizes for the training and validation data loaders.

* `model.n_speakers=1`
  * The number of speakers in the data. There is only 1 for now, but we will revisit this parameter later in the notebook.

* `model.pitch_mean=152.3 model.pitch_std=64.0 model.pitch_fmin=30 model.pitch_fmax=512`
  * For the new speaker, we need to define new pitch hyperparameters for better audio quality.
  * These parameters work for speaker 9017 from the Hi-Fi TTS dataset.
  * If you are using a custom dataset, running the script `python <NeMo_base>/scripts/dataset_processing/tts/extract_sup_data.py manifest_filepath=<your_manifest_path>` will precalculate supplementary data and print these pitch stats.
  * fmin and fmax are hyperparameters to librosa's pyin function. We recommend tweaking these only if the speaker is in a noisy environment, such that background noise isn't predicted to be speech.

* `model.optim.lr=2e-4 ~model.optim.sched model.optim.name=adam`
  * For fine-tuning, we lower the learning rate.
  * We use a fixed learning rate of 2e-4.
  * We switch from the lamb optimizer to the adam optimizer.

* `trainer.devices=1 trainer.strategy=null`
  * For this notebook, we default to 1 gpu which means that we do not need ddp.
  * If you have the compute resources, feel free to scale this up to the number of free gpus you have available.
  * Please remove the `trainer.strategy=null` section if you intend on multi-gpu training.