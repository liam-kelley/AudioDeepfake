import os
import json
import numpy as np
import torch
import soundfile as sf
import string
from pathlib import Path
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.torch.helpers import BetaBinomialInterpolator
import librosa

from get_ckpt import get_ckpt_from_last_run

def get_manifest(manifest_path):
    manifest = []
    with open(manifest_path, "r") as f:
        for i, line in enumerate(f):
            manifest.append(json.loads(line))

def load_wav(audio_file, target_sr=None):
    with sf.SoundFile(audio_file, 'r') as f:
        samples = f.read(dtype='float32')
        sample_rate = f.samplerate
        if target_sr is not None and target_sr != sample_rate:
            samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
    return samples.transpose()

def gen_mels_and_new_manifest_from_manifest(manifest, spec_model, beta_binomial_interpolator, save_dir):
    device = spec_model.device
    '''Generate a spectrograms (we need to use ground truth alignment for correct matching between audio and mels)'''
    for i, manifest_line in enumerate(manifest):
        audio = load_wav(manifest_line["audio_filepath"])
        audio = torch.from_numpy(audio).unsqueeze(0).to(device)
        audio_len = torch.tensor(audio.shape[1], dtype=torch.long, device=device).unsqueeze(0)

        #load speaker data.
        if spec_model.fastpitch.speaker_emb is not None and "speaker" in manifest_line:
            speaker = torch.tensor([manifest_line['speaker']]).to(device)
        else:
            speaker = None
        
        with torch.no_grad():
            if "normalized_text" in manifest_line:
                text = spec_model.parse(manifest_line["normalized_text"], normalize=False)
            else:
                text = spec_model.parse(manifest_line['text'])
            
            text_len = torch.tensor(text.shape[-1], dtype=torch.long, device=device).unsqueeze(0)
        
            spect, spect_len = spec_model.preprocessor(input_signal=audio, length=audio_len)

            # Generate attention prior and spectrogram inputs for HiFi-GAN
            attn_prior = torch.from_numpy(
              beta_binomial_interpolator(spect_len.item(), text_len.item())
            ).unsqueeze(0).to(text.device)
                
            spectrogram = spec_model.forward(
              text=text, 
              input_lens=text_len, 
              spec=spect, 
              mel_lens=spect_len, 
              attn_prior=attn_prior,
              speaker=speaker,
            )[0]
            
            save_path = save_dir / f"mel_{i}.npy"
            np.save(save_path, spectrogram[0].to('cpu').numpy())
            manifest_line["mel_filepath"] = str(save_path).replace(os.sep, "/")
    return manifest

def write_new_manifest(manifest,manifest_path):
    with open(manifest_path, "w") as f:
        for manifest_line in manifest:
            f.write(json.dumps(manifest_line) + '\n')

beta_binomial_interpolator = BetaBinomialInterpolator()

ckpt=get_ckpt_from_last_run(exp_manager="fastpitch_cruisetuningv2", model_name="FastPitch",get="last")
spec_model = FastPitchModel.load_from_checkpoint(ckpt)
print("FastPitch checkpoint loaded: ", ckpt)
spec_model.eval()

save_dir = Path("./cruise_mels")
save_dir.mkdir(exist_ok=True, parents=True)

manifest_train_cruise = get_manifest(manifest_path = Path("./manifests/manifest_train_cruise_only.json"))
manifest_train_hifigan = gen_mels_and_new_manifest_from_manifest(manifest_train_cruise, spec_model, beta_binomial_interpolator, save_dir)
hifigan_manifest_path = Path("./manifests/manifest_hifigan_train.json")
write_new_manifest(manifest_train_hifigan,hifigan_manifest_path)

manifest_val = get_manifest(manifest_path = Path("./manifests/manifest_val.json"))
manifest_val_hifigan = gen_mels_and_new_manifest_from_manifest(manifest_val, spec_model, beta_binomial_interpolator, save_dir)
hifigan_manifest_path = Path("./manifests/manifest_hifigan_val.json")
write_new_manifest(manifest_val_hifigan,hifigan_manifest_path)






           
