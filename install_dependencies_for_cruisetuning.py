import os

## Install dependencies
os.system('apt-get install sox libsndfile1 ffmpeg;')
#os.system('pip install Cython')
os.system('pip install wget text-unidecode pynini==2.1.4')

# ## Install NeMo
# BRANCH = 'main'
# !python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH;#egg=nemo_toolkit[all]
BRANCH = 'r1.14.0'
os.system('python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]')
os.system('pip install torchaudio>=0.10.0 -f https://download.pytorch.org/whl/torch_stable.html')

# ## Install pynini
# !wget https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo_text_processing/install_pynini.sh;
# !bash install_pynini.sh;

os.system('pip install pynini')

os.system('pip install omegaconf')
