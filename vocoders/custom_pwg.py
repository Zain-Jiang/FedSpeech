import glob
import re

import torch
from datasets.tts.utils import process_utterance
from utils.hparams import hparams
from utils.pwg_decode_from_mel import load_pwg_model
from utils.world_utils import f0_to_coarse
from vocoders.base_vocoder import BaseVocoder, register_vocoder
import numpy as np


@register_vocoder
class CustomPWG(BaseVocoder):
    def __init__(self):
        base_dir = hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
        lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0]))[-1]
        print('| load wavegan: ', ckpt)
        self.model, _, self.config, self.device = load_pwg_model(
            config_path=config_path,
            checkpoint_path=ckpt,
            stats_path=f'{base_dir}/stats.h5',
        )

    def mel2wav(self, mel, **kwargs):
        # start generation
        config = self.config
        device = self.device
        pad_size = (config["generator_params"]["aux_context_window"],
                    config["generator_params"]["aux_context_window"])
        c = mel
        with torch.no_grad():
            z = torch.randn(1, 1, c.shape[0] * config["hop_size"]).to(device)
            c = np.pad(c, (pad_size, (0, 0)), "edge")
            c = torch.FloatTensor(c).unsqueeze(0).transpose(2, 1).to(device)
            p = kwargs.get('f0')
            if p is not None:
                p[p == 1] = 0
                p = f0_to_coarse(p) + 1
                p = np.pad(p, (pad_size,), "edge")
                p = torch.LongTensor(p[None, :]).to(device)
            y = self.model(z, c, p).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out

    @staticmethod
    def wav2mel(wav_fn):
        wav_data, mel = process_utterance(
            wav_fn, fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'],
            min_level_db=hparams['min_level_db'],
            return_linear=False, vocoder='pwg')
        mel = mel.T  # [T, 80]
        return wav_data, mel
