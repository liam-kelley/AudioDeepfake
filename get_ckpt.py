from pathlib import Path

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