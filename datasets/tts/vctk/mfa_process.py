import os

os.environ["OMP_NUM_THREADS"] = "1"

import glob
from datasets.tts.base_mfa_processor import BaseMfaProcessor


class VCTKMfaProcessor(BaseMfaProcessor):

    @property
    def basedir(self):
        return 'data/raw/VCTK-Corpus'

    def meta_data(self):
        wav_fns = glob.glob(f'{self.basedir}/wav48/*/*.wav')
        for wav_fn in wav_fns:
            item_name = os.path.basename(wav_fn)[:-4]
            txt_fn = wav_fn.split("/")
            txt_fn[-1] = f'{item_name}.txt'
            txt_fn[-3] = f'txt'
            txt_fn = "/".join(txt_fn)
            if os.path.exists(txt_fn) and os.path.exists(wav_fn):
                yield item_name, wav_fn, txt_fn

    @staticmethod
    def process_txt(txt_or_fn):
        l = open(txt_or_fn).readlines()[0].strip()
        return l


if __name__ == "__main__":
    VCTKMfaProcessor().process()
