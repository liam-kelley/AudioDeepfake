import torch
import soundfile as sf
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.tts.models import FastPitchModel
from pathlib import Path

from get_ckpt import get_ckpt_from_last_run

fastpitch_from_pretrained=False
hifigan_from_pretrained=False
texts_to_say= [
    "I think it's... Really just as an actor",
    "Very fun... Uh... Character to play. challenging to play and I make enormously entertaining for an audience.",
    "The movie opens almost at the end",
    "of a movie.", 
    "And... this woman who helps him... uhhh and helps these other people and it starts this relationship between them",
    "just a dialogue and she actually has... his old job.",
    "at the hundred and tenth... and... he is intrigued by her voice and by... her abilities... and goes back",
    "basically to Washington to take her out to dinner"
]

outfilepath=Path("inferred_audios/cruisefake_v1_testing_inference2")
outfilepath.mkdir(exist_ok=True, parents=True)

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

#######################
###   LOAD MODELS   ###
#######################

if fastpitch_from_pretrained :
    spec_model = FastPitchModel.from_pretrained("tts_en_fastpitch")
else:
    ckpt=get_ckpt_from_last_run(exp_manager="fastpitch_cruisetuningv2", model_name="FastPitch",get="last")
    spec_model = FastPitchModel.load_from_checkpoint(ckpt)
    print("FastPitch checkpoint loaded: ", ckpt)
spec_model.eval().cuda()

if hifigan_from_pretrained :
    vocoder = HifiGanModel.from_pretrained("tts_hifigan")
else:
    ckpt=get_ckpt_from_last_run(exp_manager="hifigan_cruisetuningv1", model_name= "HifiGan", get="last")
    vocoder = HifiGanModel.load_from_checkpoint(ckpt)
    print("HifiGan checkpoint loaded: ", ckpt)
vocoder = vocoder.eval().cuda()

#######################
###    INFERENCE    ###
#######################

for i,text in enumerate(texts_to_say):
    spec, audio = infer(spec_model, vocoder, text, speaker=1)
    # Save the audio to disk in a file called speech.wav
    sf.write(outfilepath / Path("speech" + str(i) + ".wav"), audio[0], 22050)
    print("Audio ", "speech" + str(i) + ".wav", " inferred and written.")
