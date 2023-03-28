import matplotlib.pyplot as plt

loss_files=[
    "exp/fastpitch_cruisetuningv2/FastPitch/2023-03-23_02-40-51/lightning_logs.txt",
    "exp/hifigan_cruisetuningv1/HifiGan/2023-03-23_16-31-36/lightning_logs.txt"
]

for loss_file in loss_files:
    print(loss_file)
    dico={"epoch":[], "val_loss" : []}
    with open(loss_file, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            if line[:5] != "Epoch":
                continue
            split_lines=line.strip().split()
            if split_lines[7][:-1] != "no":
                dico["epoch"].append(float(split_lines[1][:-1]))
                dico["val_loss"].append(float(split_lines[7][:-1]))
    print(len(dico["epoch"]))
    plt.plot(dico["epoch"],dico["val_loss"])
    plt.show()
