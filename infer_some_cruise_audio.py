import torch
import soundfile as sf
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.tts.models import FastPitchModel

hifigan_from_pretrained=True

def infer(spec_gen_model, vocoder_model, str_input, speaker=None):
    """
    Synthesizes spectrogram and audio from a text string given a spectrogram synthesis and vocoder model.
    
    Args:
        spec_gen_model: Spectrogram generator model (FastPitch in our case)
        vocoder_model: Vocoder model (HiFiGAN in our case)
        str_input: Text input for the synthesis
        speaker: Speaker ID
    
    Returns:
        spectrogram and waveform of the synthesized audio.
    """
    with torch.no_grad():
        parsed = spec_gen_model.parse(str_input)
        if speaker is not None:
            speaker = torch.tensor([speaker]).long().to(device=spec_gen_model.device)
        spectrogram = spec_gen_model.generate_spectrogram(tokens=parsed, speaker=speaker)
        audio = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)
        
    if spectrogram is not None:
        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.to('cpu').numpy()
        if len(spectrogram.shape) == 3:
            spectrogram = spectrogram[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.to('cpu').numpy()
    return spectrogram, audio

if hifigan_from_pretrained :
    vocoder = HifiGanModel.from_pretrained("tts_hifigan")
vocoder = vocoder.eval().cuda()

ckpt = "cruisetuningv2/FastPitch/2023-03-20_21-48-53/checkpoints/FastPitch--val_loss=1.3965-epoch=130-last.ckpt"
spec_model = FastPitchModel.load_from_checkpoint(ckpt)
spec_model.eval().cuda()

text_to_say="My name is tom cruise. How are you? I was born in a small town. There were about thirty people. They were all very Nice."

spec, audio = infer(spec_model, vocoder, text_to_say, speaker=1)
# Save the audio to disk in a file called speech.wav
sf.write("speech.wav", audio[0], 22050)
