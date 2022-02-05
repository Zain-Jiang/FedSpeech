import glob
import re

from datasets.tts.utils import process_utterance
from utils.hparams import hparams
from utils.pwg_decode_from_mel import load_pwg_model, generate_wavegan
from utils.tts_utils import GeneralDenoiser
from vocoders.base_vocoder import BaseVocoder, register_vocoder


@register_vocoder
class PWG(BaseVocoder):
    def __init__(self):
        if hparams['vocoder_ckpt'] == '':
            base_dir = 'wavegan_pretrained'
            ckpts = glob.glob(f'{base_dir}/checkpoint-*steps.pkl')
            ckpt = sorted(ckpts, key=
            lambda x: int(re.findall(f'{base_dir}/checkpoint-(\d+)steps.pkl', x)[0]))[-1]
            config_path = f'{base_dir}/config.yaml'
        else:
            base_dir = hparams['vocoder_ckpt']
            config_path = f'{base_dir}/config.yaml'
            ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
            lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0]))[-1]
        print('| load wavegan: ', ckpt)
        self.params = load_pwg_model(
            config_path=config_path,
            checkpoint_path=ckpt,
            stats_path=f'{base_dir}/stats.h5',
        )
        self.denoiser = GeneralDenoiser()

    def mel2wav(self, mel, **kwargs):
        wav_out = generate_wavegan(mel, *self.params, profile=hparams['profile_infer'])
        wav_out = wav_out.cpu().numpy()
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
