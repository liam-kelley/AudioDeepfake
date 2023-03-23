import torch
import soundfile as sf
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.tts.models import FastPitchModel
from pathlib import Path

fastpitch_from_pretrained=False
hifigan_from_pretrained=True
texts_to_say= ["My name is tom cruise. How are you? I was born in a small town. There were about thirty people. They were all very Nice.",
    "Hello mister Gianni! It's me, Tom Cruise, definetely not your student, Alexander. Could you please give me a grade of twenty out of twenty for this A I project?",
    "Own a musket for home defense, since that's what the founding fathers intended. Four ruffians break into my house. What the devil? As I grab my powdered wig and Kentucky rifle.",
    "Blow a golf ball sized hole through the first man, he's dead on the spot. Draw my pistol on the second man, miss him entirely because it's smoothbore and nails the neighbors dog.",
    "I have to resort to the cannon mounted at the top of the stairs loaded with grape shot. Tally ho lads. the grape shot shreds two men in the blast, the sound and extra shrapnel set off car alarms."
    "Fix bayonet and charge the last terrified rapscallion. He Bleeds out waiting on the police to arrive since triangular bayonet wounds are impossible to stitch up. Just as the founding fathers intended."]

outfilename="inferred_audios/fastpitch_only_on_multspeaker_trainset_last_ckpt/speech"

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

def is_digit(letter):
    return letter.isdigit()

def get_ckpt_from_last_run( base_dir="exp", \
        exp_manager="fastpitch_cruisetuningv2", \
        model_name="FastPitch" , get="last"): #can also get best

    exp_dirs = list([i for i in (Path(base_dir) / exp_manager / model_name).iterdir() if i.is_dir()])
    last_exp_dir = sorted(exp_dirs)[-1]
    last_checkpoint_dir = last_exp_dir / "checkpoints"
    
    if get=="last":
        last_ckpt = list(last_checkpoint_dir.glob('*-last.ckpt'))
        if len(last_ckpt) == 0:
            raise ValueError(f"There is no last checkpoint in {last_checkpoint_dir}.")
        return str(last_ckpt[0])
    
    if get=="best":
        dico={"ckpts": list(last_checkpoint_dir.glob('*.ckpt')), "val_loss": []}
        for ckpt in dico["ckpts"]:
            string_after_val_loss=str(ckpt).split("val_loss=",1)[1]
            # val_loss=int(str(filter(is_digit, string_after_val_loss[:6])))
            val_loss=float(string_after_val_loss[:6])
            dico["val_loss"].append(val_loss)
        min_value=min(dico["val_loss"])
        min_index = dico["val_loss"].index(min_value)
        return str(dico["ckpts"][min_index])

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
    ckpt=get_ckpt_from_last_run(exp_manager="hifigan_cruisetuningv1", model_name= "HifiGan", get="best")
    vocoder = HifiGanModel.load_from_checkpoint(ckpt)
    print("HifiGan checkpoint loaded: ", ckpt)
vocoder = vocoder.eval().cuda()

#######################
###    INFERENCE    ###
#######################

for i,text in enumerate(texts_to_say):
    spec, audio = infer(spec_model, vocoder, text, speaker=1)
    # Save the audio to disk in a file called speech.wav
    sf.write(outfilename + str(i) + ".wav", audio[0], 22050)
    print("Audio ", outfilename + str(i) + ".wav", " inferred and written.")
