import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

from utils.pl_utils import data_loader
import shutil, os, sys, copy
from multiprocessing.pool import Pool
from tqdm import tqdm

from modules.tts_modules import DurationPredictorLoss
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset
from utils.plot import plot_to_figure
from utils.world_utils import restore_f0, process_f0

import numpy as np

from modules.fs2 import FastSpeech2
from tasks.fs2 import FastSpeechDataset, FastSpeech2Task
from tasks.transformer_tts import TransformerTtsTask
from tasks.base_task import BaseDataset

import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import utils
from utils import audio


class FastSpeechDataset_FedAVG(BaseDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self, data_dir, phone_encoder, prefix, hparams, shuffle=False):
        super().__init__(data_dir, prefix, hparams, shuffle)
        self.phone_encoder = phone_encoder
        self.data = None
        self.idx2key = np.load(f'{self.data_dir}/{self.prefix}_all_keys.npy')
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.indexed_bs = None

        # pitch stats
        self.f0_mean, self.f0_std = np.load(f'{self.data_dir}/train_f0s_mean_std.npy')
        hparams['f0_mean'] = self.f0_mean
        hparams['f0_std'] = self.f0_std

        self.last_steps = 0
        self.global_steps = 0
        self.spk_step = 0
        self.switch_spk = False
        self.average = False

    def _get_item(self, index):
        if self.indexed_bs is None:
            self.indexed_bs = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_bs[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        spec = torch.Tensor(item['mel'])
        energy = (spec.exp() ** 2).sum(-1).sqrt()[:hparams['max_frames']]
        mel2ph = torch.LongTensor(item['mel2ph'])[:hparams['max_frames']]
        f0, uv = process_f0(item["f0"], hparams)
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])

        sample = {
            "id": index,
            "utt_id": item['item_name'],
            "text": item['txt'],
            "source": phone,
            "target": spec[:hparams['max_frames']],
            "pitch": torch.LongTensor(item.get("pitch"))[:hparams['max_frames']],
            "energy": energy,
            "f0": f0[:hparams['max_frames']],
            "uv": uv[:hparams['max_frames']],
            "mel2ph": mel2ph,
        }
        if self.hparams['use_spk_embed']:
            sample["spk_id"] = item['spk_id']
            sample["spk_embed"] = item['spk_embed']
        return sample

    def switch_state(self):

        # switch spk
        if self.global_steps - self.last_steps >= 100:
            self.last_steps = self.global_steps
            self.spk_step = (self.spk_step + 1) % 10

            # print("switch spk: " + str(self.spk_step + 3))
            self.switch_spk = True

            # Averaging
            if self.spk_step == 0:
                print("Average model")
                self.average = True

        self.global_steps += 1
        return self.spk_step + 3

    def collater(self, samples):
        samples_copy = []
        spk_step = self.switch_state()
        for index, s in enumerate(samples):
            if spk_step == s['spk_id']:
                samples_copy.append(s)
        samples = samples_copy

        if len(samples) == 0:
            return {}
        pad_idx = self.phone_encoder.pad()
        id = torch.LongTensor([s['id'] for s in samples])
        utt_ids = [s['utt_id'] for s in samples]
        text = [s['text'] for s in samples]

        src_tokens = utils.collate_1d([s['source'] for s in samples], pad_idx)
        f0 = utils.collate_1d([s['f0'] for s in samples], -200) if self.hparams['use_pitch_embed'] else None
        pitch = utils.collate_1d([s['pitch'] for s in samples]) if self.hparams[
            'use_pitch_embed'] else None
        uv = utils.collate_1d([s['uv'] for s in samples]) if self.hparams['use_pitch_embed'] else None
        energy = utils.collate_1d([s['energy'] for s in samples], pad_idx) if self.hparams['use_energy_embed'] else None
        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], pad_idx)
        target = utils.collate_2d([s['target'] for s in samples], pad_idx)
        prev_output_mels = utils.collate_2d([s['target'] for s in samples], pad_idx, shift_right=True)

        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        target_lengths = torch.LongTensor([s['target'].shape[0] for s in samples])
        ntokens = sum(len(s['source']) for s in samples)
        nmels = sum(len(s['target']) for s in samples)

        batch = {
            'id': id,
            'utt_id': utt_ids,
            'nsamples': len(samples),
            'ntokens': ntokens,
            'nmels': nmels,
            'text': text,
            'src_tokens': src_tokens,
            'mel2ph': mel2ph,
            'src_lengths': src_lengths,
            'targets': target,
            'energy': energy,
            'target_lengths': target_lengths,
            'prev_output_mels': prev_output_mels,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        }

        if self.hparams['use_spk_embed']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            spk_embed = torch.FloatTensor([s['spk_embed'] for s in samples])
            batch['spk_ids'] = spk_ids
            batch['spk_embed'] = spk_embed
        return batch

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', 0))


class Fs2TransferTask(FastSpeech2Task):
    def __init__(self):
        super(Fs2TransferTask, self).__init__()
        self.dur_loss_fn = DurationPredictorLoss()
        self.mse_loss_fn = torch.nn.MSELoss()
        self.global_parameters = {}
        self.sum_parameters = None

    def get_mask(self, dataset, spk_ids):
        mask = torch.zeros(len(dataset))
        for index, item in enumerate(dataset):
            # if item['spk_id'] not in spk_ids:
            #     mask[index] = 1
            for spk_id in spk_ids:
                if item['spk_id'] == spk_id:
                    mask[index] = 1
        return mask

    @data_loader
    def train_dataloader(self):
        self.train_dataset = FastSpeechDataset_FedAVG(hparams['data_dir'], self.phone_encoder,
                                                      hparams['train_set_name'], hparams, shuffle=True)
        spk_mask = None
        for id in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            mask = self.get_mask(self.train_dataset, [id])
            idx_mask = np.where(mask == 1)[0]
            rand_mask = np.random.choice(idx_mask, len(idx_mask) - 100, replace=False)
            mask[rand_mask] = 0
            if spk_mask == None:
                spk_mask = mask
            else:
                spk_mask[np.where(mask == 1)[0]] = 1

        return self.build_dataloader(self.train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'], spk_mask=spk_mask)

    @data_loader
    def val_dataloader(self):
        valid_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['valid_set_name'], hparams,
                                          shuffle=False)
        spk_mask = self.get_mask(valid_dataset, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        return self.build_dataloader(valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences,
                                     spk_mask=spk_mask)

    @data_loader
    def test_dataloader(self):
        test_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                         hparams['test_set_name'], hparams, shuffle=False)
        spk_mask = self.get_mask(test_dataset, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        return self.build_dataloader(test_dataset, False, self.max_eval_tokens, self.max_eval_sentences,
                                     spk_mask=spk_mask)

    def validation_end(self, outputs):
        loss_output = self._validation_end(outputs)
        print(f"\n==============\n "
              f"valid results: {loss_output}\n"
              f"\n==============\n")
        with open("valid_loss_baseline_FedAVG.txt", 'a+') as file_object:
            file_object.write(f"valid results: {loss_output}\n")
        return {
            'log': {f'val/{k}': v for k, v in loss_output.items()},
            'val_loss': loss_output['total_loss']
        }

    def on_sanity_check_start(self):
        self.global_parameters = self.model.state_dict()

    def _training_step(self, sample, batch_idx, _):
        #print(self.train_dataset.global_steps)

        if self.train_dataset.switch_spk:
            if self.sum_parameters == None:
                self.sum_parameters = self.model.state_dict()
            else:
                for item in self.sum_parameters:
                    self.sum_parameters[item] = self.sum_parameters[item] + self.model.state_dict()[item]
            self.model.load_state_dict(self.global_parameters, strict=True)
            self.train_dataset.switch_spk = False

        if self.train_dataset.average:
            for item in self.global_parameters:
                self.global_parameters[item] = (self.sum_parameters[item] / 10)
            self.model.load_state_dict(self.global_parameters, strict=True)
            self.sum_parameters = None
            self.train_dataset.average = False

        if sample == {}:
            loss_output = {'mel': torch.tensor([0.], requires_grad=True).cuda(),
                           'dur': torch.tensor([0.], requires_grad=True).cuda(),
                           'uv': torch.tensor([0.], requires_grad=True).cuda(),
                           'f0': torch.tensor([0.], requires_grad=True).cuda()}
            total_loss = sum([v for v in loss_output.values() if v.requires_grad])
            loss_output['batch_size'] = torch.tensor([0.], requires_grad=True).cuda()
            return total_loss, loss_output
        else:
            #print(sample['spk_ids'])
            loss_output = self.run_model(self.model, sample)
            total_loss = sum([v for v in loss_output.values() if v.requires_grad])
            loss_output['batch_size'] = sample['targets'].size()[0]
            return total_loss, loss_output

    def test_step(self, sample, batch_idx):
        """Performs evaluation."""

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        src_tokens = sample['src_tokens']

        if hparams['profile_infer']:
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            mel2ph = sample['mel2ph']
            f0 = sample['f0']
            uv = sample['uv']
        else:
            mel2ph = None
            f0 = None
            uv = None

        ref_mels = sample['targets'][:1].repeat([sample['targets'].shape[0], 1, 1]) if hparams['use_ref_enc'] else None
        with utils.Timer('fs', print_time=hparams['profile_infer']):
            outputs = self.model(src_tokens, mel2ph=mel2ph, spk_embed=spk_embed, f0=f0, uv=uv, ref_mels=ref_mels)

        sample['outputs'] = outputs['mel_out']
        sample['mel2ph_pred'] = outputs['mel2ph']
        sample['f0'] = restore_f0(sample['f0'], uv if hparams['use_uv'] else None, hparams)
        sample['f0_pred'] = outputs.get('f0')

        return self.after_infer(sample)

    @staticmethod
    def save_result(wav_out, mel, prefix, utt_id, text, gen_dir,
                    pitch=None, alignment=None, str_phs=None, mel2ph=None, spk_ids=None):
        base_fn = f'[{prefix}][{utt_id}]'
        base_fn += text.replace(":", "%3A")[:80]
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{spk_ids}/{base_fn}.wav', hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])

        fig = plt.figure(figsize=(14, 10))
        heatmap = plt.pcolor(mel.T)
        fig.colorbar(heatmap)
        if pitch is not None:
            if isinstance(pitch, list):
                for x_ in pitch:
                    plt.plot(x_ / 10, color='w')
            else:
                plt.plot(pitch / 10, color='w')
        if mel2ph is not None:
            decoded_txt = str_phs.split(" ")
            ax = plt.gca()
            ax.set_yticks(np.arange(len(decoded_txt)))
            ax.set_yticklabels(list(decoded_txt), fontsize=8)
            plt.plot(mel2ph, color='r')

        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png')
        plt.close(fig)

        if alignment is not None:
            fig, ax = plt.subplots(figsize=(12, 16))
            im = ax.imshow(alignment, aspect='auto', origin='lower',
                           interpolation='none')
            decoded_txt = str_phs.split(" ")
            ax.set_yticks(np.arange(len(decoded_txt)))
            ax.set_yticklabels(list(decoded_txt), fontsize=6)
            fig.colorbar(im, ax=ax)
            fig.savefig(f'{gen_dir}/attn_plot/{base_fn}_attn.png', format='png')
            plt.close()

    def after_infer(self, predictions):
        if self.saving_result_pool is None and not hparams['profile_infer']:
            self.saving_result_pool = Pool(8)
            self.saving_results_futures = []
        predictions = utils.unpack_dict_to_list(predictions)
        t = tqdm(predictions)
        for num_predictions, prediction in enumerate(t):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            utt_id = prediction.get('utt_id')
            text = prediction.get('text')
            targets = self.remove_padding(prediction.get("targets"))
            outputs = self.remove_padding(prediction["outputs"])
            mel2ph_pred = self.remove_padding(prediction.get("mel2ph_pred"))
            mel2ph_gt = self.remove_padding(prediction.get("mel2ph"))
            f0_pred = self.remove_padding(prediction.get("f0_pred"))
            f0_gt = self.remove_padding(prediction.get("f0"))
            str_phs = self.phone_encoder.decode(prediction.get('src_tokens'), strip_padding=True)
            # print(outputs.shape, f0_pred.shape)
            spk_ids = prediction.get("spk_ids")

            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
            wav_pred = self.vocoder.mel2wav(outputs, f0=f0_pred)
            if not hparams['profile_infer']:
                os.makedirs(gen_dir, exist_ok=True)
                os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
                os.makedirs(f'{gen_dir}/plot', exist_ok=True)
                for i in range(1, 13):
                    os.makedirs(f'{gen_dir}/wavs/{i}', exist_ok=True)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_pred, outputs, f'P', utt_id, text, gen_dir, f0_pred, None,
                        str_phs, mel2ph_pred, spk_ids]))

                if targets is not None and hparams['save_gt']:
                    wav_gt = self.vocoder.mel2wav(targets, f0=f0_gt)
                    self.saving_results_futures.append(
                        self.saving_result_pool.apply_async(self.save_result, args=[
                            wav_gt, targets, 'G', utt_id, text, gen_dir, f0_gt, None,
                            str_phs, mel2ph_gt, spk_ids]))
                t.set_description(
                    f"Pred_shape: {outputs.shape}, gt_shape: {targets.shape}")
            else:
                if 'gen_wav_time' not in self.stats:
                    self.stats['gen_wav_time'] = 0
                self.stats['gen_wav_time'] += len(wav_pred) / hparams['audio_sample_rate']
                print('gen_wav_time: ', self.stats['gen_wav_time'])

        return {}


if __name__ == '__main__':
    hparams['save_gt'] = True
    Fs2TransferTask.start()
