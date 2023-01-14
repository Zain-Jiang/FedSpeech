# FedSpeech 👋
This is the official implementation for the paper "FedSpeech: Federated Text-to-Speech with Continual Learning"

## Content
1. Implementation of FedSpeech
2. Implementation of all baselines in the paper.
3. Sample Audio files for FedSpeech and all baselines can be found in ./sample_audio

##Reproducibility
1. Our random seed: 1234
2. The specification of dependencies is described in the requirements.txt
3. The Training time is about 14 hours and the inference time is about 10 minutes.

## Install Dependencies
Operation system: Ubuntu 18.04.1 LTS (GNU/Linux 4.15.0-29-generic x86_64)
GPU memory: at least 11GB
```bash
export PYTHONPATH=.
# build a virtual env (recommended)
python -m venv venv
source venv/bin/activate
# install requirements
pip install -r requirements.txt
```

## Install MFA
```bash
bash scripts/install_mfa.sh
```

##  Prepare Dataset
#### Build *VCTK* dataset
- Download VCTK from https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html and put `VCTK-Corpus` to `data/raw/VCTK-Corpus`

- Prepare MFA inputs
```bash
python datasets/tts/vctk/mfa_process.py
```

- Forced alignment
```bash
export MFA_CORPUS=VCTK-Corpus; ./mfa/dist/montreal-forced-aligner/lib/train_and_align data/raw/$MFA_CORPUS/mfa_input data/raw/$MFA_CORPUS/dict_mfa.txt data/raw/$MFA_CORPUS/mfa_outputs -t ./montreal-forced-aligner/tmp -j 24
```

#### Build binary data
```bash
# fs2
python datasets/tts/vctk/gen_fs2.py --config configs/tts/vctk/fs2.yaml
```

##  Prepare Pretrained Vocoder
```bash
mkdir checkpoints/0717_vctkpwgk3_1
```
- Download https://fedspeech.github.io/FedSpeech_example/0717_vctkpwgk3_1/config.yaml and https://fedspeech.github.io/FedSpeech_example/0717_vctkpwgk3_1/model_ckpt_steps_381000.ckpt Put them into checkpoints/0717_vctkpwgk3_1.

## Run FedSpeech
```bash
#start first round training
scripts/run_FedSpeech.sh first_round

#start second round training
scripts/run_FedSpeech.sh second_round
```

## Run Baselines
- Scratch
```bash
scripts/run_baseline_scratch_100.sh
```
- Finetune
```bash
scripts/run_baseline_finetune_100.sh
```
- Multitask
```bash
scripts/run_baseline_multitask_100.sh
```
- FedAvg
```bash
scripts/run_baseline_FedAVG.sh
```
- CPG
```bash
scripts/run_baseline_CPG.sh
```

