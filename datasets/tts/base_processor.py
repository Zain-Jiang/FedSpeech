import glob
import json
import os
import traceback
from multiprocessing.pool import Pool

from resemblyzer import VoiceEncoder
from tqdm import tqdm
from datasets.tts.utils import get_mel2ph, get_pitch, build_phone_encoder
from utils.hparams import set_hparams, hparams
import numpy as np

from utils.indexed_datasets import IndexedDatasetBuilder
from vocoders.base_vocoder import VOCODERS


class BaseProcessor:
    def __init__(self):
        set_hparams()
        raw_data_dir = hparams['raw_data_dir']
        self.save_wav = hparams['with_wav']
        self.item2txt = {os.path.splitext(os.path.basename(v))[0]: v
                         for v in glob.glob(f"{raw_data_dir}/mfa_input/*/*.text")}
        self.item2tgfn = {os.path.splitext(os.path.basename(v))[0]: v
                          for v in glob.glob(f"{raw_data_dir}/mfa_outputs/*/*.TextGrid")}
        self.item_names = sorted(list(self.item2txt.keys()))

    @property
    def train_item_names(self):
        raise NotImplementedError

    @property
    def valid_item_names(self):
        raise NotImplementedError

    @property
    def test_item_names(self):
        return self.valid_item_names

    def item_name2spk_id(self, item_name):
        return 0

    def _phone_encoder(self):
        raw_data_dir = hparams['raw_data_dir']
        ph_set = [x.split(' ')[0] for x in open(f'{raw_data_dir}/dict.txt').readlines()]
        json.dump(ph_set, open(f"{hparams['data_dir']}/phone_set.json", 'w'))
        print("| phone set: ", ph_set)
        return build_phone_encoder(hparams['data_dir'])

    def meta_data(self, prefix):
        raw_data_dir = hparams['raw_data_dir']
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            if item_name not in self.item2tgfn:
                print(f"| Textgrid not found. Skip {item_name}.")
                continue
            tg_fn = self.item2tgfn[item_name]
            group_id = tg_fn.split("/")[-2]
            ph_fn = f"{raw_data_dir}/mfa_input/{group_id}/{item_name}.ph"
            wav_fn = f"{raw_data_dir}/mfa_input/{group_id}/{item_name}.wav"
            txt = self.item2txt[item_name]
            spk_id = self.item_name2spk_id(item_name)
            yield item_name, ph_fn, txt, tg_fn, wav_fn, spk_id

    def process(self):
        os.makedirs(hparams['data_dir'], exist_ok=True)
        self.phone_encoder = self._phone_encoder()
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def process_data(self, prefix):
        data_dir = hparams['data_dir']
        futures = []
        p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
        for m in self.meta_data(prefix):
            item_name, ph_fn, txt_fn, tg_fn, wav_fn, spk_id = m
            futures.append([
                m, p.apply_async(self.process_item, args=(
                    ph_fn, txt_fn, tg_fn, wav_fn, self.phone_encoder, hparams))])
        p.close()

        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        all_keys = []
        lengths = []
        f0s = []
        durs = []
        total_sec = 0
        voice_encoder = VoiceEncoder().cuda() if hparams['use_spk_embed'] else None

        for future in tqdm(futures):
            inp = future[0]
            res = future[1].get()
            if res is None:
                continue
            phone_encoded, txt, mel, mel2ph, dur, f0, pitch_coarse, wav_data = res
            assert mel2ph.max() - 1 < len(mel)
            item_name, spk_id = inp[0], inp[5]
            if sum(f0) == 0:
                print(f"| Skip {item_name}. ({txt})")
                continue
            spk_embed = voice_encoder.embed_utterance(wav_data) if hparams['use_spk_embed'] else None
            item = {
                'item_name': item_name,
                'txt': txt,
                'phone': phone_encoded,
                'mel': mel,
                'mel2ph': mel2ph,
                'spk_id': spk_id,
                'spk_embed': spk_embed,
                'pitch': pitch_coarse,
                'f0': f0,
            }
            if self.save_wav:
                item['wav'] = wav_data
            builder.add_item(item)
            lengths.append(mel.shape[0])
            all_keys.append(item_name)
            f0s.append(f0)
            durs.append(dur)
            total_sec += len(wav_data) / hparams['audio_sample_rate']
        p.join()
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_all_keys.npy', all_keys)
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        f0s = np.concatenate(f0s, 0)
        f0s = f0s[f0s != 0]
        np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f"| {prefix} total duration: {total_sec:.3f}s.")

    @staticmethod
    def process_item(ph_fn, txt_fn, tg_fn, wav_fn, encoder, hparams):
        ph = open(ph_fn).readlines()[0].strip()
        txt = open(txt_fn).readlines()[0].strip()
        ph = "| " + ph + " |"
        try:
            phone_encoded = encoder.encode(ph)
            wav_data, mel = VOCODERS[hparams['vocoder']].wav2mel(wav_fn)
        except:
            traceback.print_exc()
            print("| invalid data", wav_fn)
            return None

        mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hparams)
        f0, pitch_coarse = get_pitch(wav_data, mel, hparams)
        return phone_encoded, txt, mel, mel2ph, dur, f0, pitch_coarse, wav_data
