import os

os.environ["OMP_NUM_THREADS"] = "1"

import glob
import traceback
import json
import os
import re
import subprocess
from multiprocessing.pool import Pool
from tqdm import tqdm
import pandas as pd

g2p = None
puncs = '!,.?;'
try:
    from g2p_en import G2p
    from builtins import str as unicode
    import unicodedata
    from g2p_en.expand import normalize_numbers
    from nltk import word_tokenize, pos_tag


    class MyG2p(G2p):
        def __call__(self, text):
            # preprocessing
            text = unicode(text)
            text = normalize_numbers(text)
            text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn')  # Strip accents
            text = text.lower()
            text = re.sub(f"[^ a-z'{puncs}]", "", text)
            text = text.replace("i.e.", "that is")
            text = text.replace("e.g.", "for example")

            # tokenization
            words = word_tokenize(text)
            tokens = pos_tag(words)  # tuples of (word, tag)

            # steps
            prons = []
            for word, pos in tokens:
                if re.search("[a-z]", word) is None:
                    pron = [word]

                elif word in self.homograph2features:  # Check homograph
                    pron1, pron2, pos1 = self.homograph2features[word]
                    if pos.startswith(pos1):
                        pron = pron1
                    else:
                        pron = pron2
                elif word in self.cmu:  # lookup CMU dict
                    pron = self.cmu[word][0]
                else:  # predict for oov
                    pron = self.predict(word)

                prons.extend(pron)
                prons.extend([" "])

            return prons[:-1]
except Exception as e:
    traceback.print_exc()

class BaseMfaProcessor:
    def __init__(self):
        global g2p
        g2p = MyG2p()

    @property
    def basedir(self):
        raise NotImplementedError

    def meta_data(self):
        raise NotImplementedError

    @staticmethod
    def process_txt(txt_or_fn):
        raise NotImplementedError

    def process(self):
        # build mfa_input for forced alignment
        basedir = self.basedir
        p = Pool(os.cpu_count())
        subprocess.check_call(f'rm -rf {basedir}/mfa_input', shell=True)
        futures = []

        for idx, (item_name, wav_fn, txt_or_fn) in enumerate(tqdm(self.meta_data())):
            futures.append([(item_name, wav_fn, txt_or_fn),
                            p.apply_async(self.g2p_job, args=[
                                idx, self.process_txt, item_name, wav_fn, txt_or_fn, basedir])])
        p.close()
        mfa_dict = set()
        phone_set = set()
        meta_df = []
        for idx, (inp_args, f) in enumerate(tqdm(futures)):
            item_name, wav_fn, txt_or_fn = inp_args
            phs, phs_mfa, group, txt, txt_raw = f.get()
            meta_df.append({
                'item_name': item_name, 'group': group,
                'txt': txt, 'txt_raw': txt_raw, 'ph': phs
            })
            for ph in phs.split(" "):
                phone_set.add(ph)
            for ph_mfa in phs_mfa.split(" "):
                mfa_dict.add(ph_mfa)
        mfa_dict = sorted(mfa_dict)
        phone_set = sorted(phone_set)
        print("| mfa dict: ", mfa_dict)
        print("| phone set: ", phone_set)
        with open(f'{basedir}/dict_mfa.txt', 'w') as f:
            for ph in mfa_dict:
                f.write(f'{ph} {ph}\n')
        with open(f'{basedir}/dict.txt', 'w') as f:
            for ph in phone_set:
                f.write(f'{ph} {ph}\n')
        phone_set = ["<pad>", "<EOS>", "<UNK>"] + phone_set
        json.dump(phone_set, open(f'{basedir}/phone_set.json', 'w'))
        p.join()

        # save to csv
        meta_df = pd.DataFrame(meta_df)
        meta_df.to_csv(f"{basedir}/metadata_phone.csv")

    @staticmethod
    def g2p_job(idx, process_txt, item_name, wav_fn, txt_or_fn, basedir):
        group = idx // 100
        txt_raw = process_txt(txt_or_fn)
        txt = re.sub("[\'\-]+", "", txt_raw)
        phs = [p.replace(" ", "|") for p in g2p(txt)]
        phs_str = " ".join(phs)
        phs_str = re.sub(f'\| ([{puncs}])( \|)?', r'\1', phs_str)
        os.makedirs(f'{basedir}/mfa_input/{group}', exist_ok=True)
        with open(f'{basedir}/mfa_input/{group}/{item_name}.text', 'w') as f_txt:
            f_txt.write(txt)
        with open(f'{basedir}/mfa_input/{group}/{item_name}.ph', 'w') as f_txt:
            phs_str = re.sub(f"([{puncs}])+", r"\1", phs_str)
            f_txt.write(phs_str)
        with open(f'{basedir}/mfa_input/{group}/{item_name}.lab', 'w') as f_txt:
            phs_str_mfa = re.sub(f"[{puncs}]+", "PUNC", phs_str)
            phs_str_mfa = re.sub(" \| ", " ", phs_str_mfa)
            f_txt.write(phs_str_mfa)
        subprocess.check_call(f'cp "{wav_fn}" "{basedir}/mfa_input/{group}/"', shell=True)
        return phs_str, phs_str_mfa, group, txt, txt_raw
