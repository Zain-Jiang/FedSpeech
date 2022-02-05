import os

os.environ["OMP_NUM_THREADS"] = "1"

from copy import deepcopy
from datasets.tts.base_processor import BaseProcessor
import random
from utils.hparams import hparams


class Fs2VCTKProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        spk_map = set()
        for item_name in self.item_names:
            spk_name = item_name.split("_")[0]
            spk_map.add(spk_name)
        self.spk_map = {x: i for i, x in enumerate(sorted(list(spk_map)))}
        print("| spk_map: ", spk_map)
        print(len(spk_map))
        assert len(spk_map) <= hparams['num_spk']
        self.item_names_shuffle = deepcopy(self.item_names)
        random.seed(1234)
        random.shuffle(self.item_names_shuffle)

    @property
    def train_item_names(self):
        return self.item_names_shuffle[4000:]

    @property
    def valid_item_names(self):
        return self.item_names_shuffle[:4000]

    def item_name2spk_id(self, item_name):
        return self.spk_map[item_name.split("_")[0]]


if __name__ == "__main__":
    Fs2VCTKProcessor().process()
