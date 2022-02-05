import os
os.environ["OMP_NUM_THREADS"] = "1"

from datasets.tts.lj.gen_fs2 import Fs2LJProcessor
from datasets.tts.utils import build_phone_encoder
import glob
import pandas as pd
from utils.hparams import hparams


class Fs2VctkAdaptLJProcessor(Fs2LJProcessor):
    @staticmethod
    def process_txt(ph_or_fn, txt_or_fn):
        ph = open(ph_or_fn).readlines()[0].strip().replace("|", "SEP")
        return ph, txt_or_fn

    def _phone_encoder(self):
        os.system(f'cp data/vctk/phone_set.json {hparams["data_dir"]}/')
        return build_phone_encoder(hparams["data_dir"])

    def meta_data(self, prefix):
        raw_data_dir = hparams['raw_data_dir']
        data_df = pd.read_csv(os.path.join(raw_data_dir, 'metadata_phone.csv'))
        fn2txt = {k: v for k, v in zip(data_df['wav'], data_df['txt1'])}
        all_wav_fns = sorted(glob.glob(f'{raw_data_dir}/wavs/*.wav'))
        item2tgfn = {os.path.splitext(os.path.basename(v))[0]: v
                     for v in glob.glob(f"{raw_data_dir}/mfa_outputs/*/*.TextGrid")}
        if prefix == 'valid' or prefix == 'test':
            all_wav_fns = all_wav_fns[:523]
        else:
            all_wav_fns = all_wav_fns[523:524]

        item_name2txt_path = glob.glob(f'{raw_data_dir}/mfa_input/*/*.ph')
        item_name2txt_path = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in item_name2txt_path
        }
        for wav_fn in all_wav_fns:
            item_name = os.path.splitext(os.path.basename(wav_fn))[0]
            ph_fn = item_name2txt_path[item_name]
            yield "lj_" + item_name, ph_fn, fn2txt[item_name], item2tgfn[item_name], wav_fn, 0


if __name__ == "__main__":
    Fs2VctkAdaptLJProcessor().process()
