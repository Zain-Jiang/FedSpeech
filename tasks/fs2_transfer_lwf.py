import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
from pytorch_lightning.logging import TensorBoardLogger
from utils.pl_utils import data_loader, DDP, BaseTrainer, LatestModelCheckpoint
import shutil, os
import copy
from multiprocessing.pool import Pool
from tqdm import tqdm

from modules.tts_modules import DurationPredictorLoss
from utils.hparams import hparams, set_hparams
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


class Fs2lwf(FastSpeech2Task):
    def __init__(self):
        super(Fs2lwf, self).__init__()
        self.dur_loss_fn = DurationPredictorLoss()
        self.mse_loss_fn = torch.nn.MSELoss()
        self.original_model_loaded = False
        self.lwf_threhold = hparams['lwf_threhold']
        # print(self.original_model)
        # for i in checkpoint:
        #     print(i)
        # self.original_model.load_state_dict(checkpoint['state_dict'])
        # if on_gpu:
        #     self.original_model.cuda(self.root_gpu)
        # # load training state (affects trainer only)
        # del checkpoint

    def get_mask(self, dataset, spk_ids):
        mask = torch.zeros(len(dataset))
        for index, item in enumerate(dataset):
            # if item['spk_id'] in spk_ids:
            #     mask[index] = 1
            for spk_id in spk_ids:
                if item['spk_id'] == spk_id:
                    mask[index] = 1
        return mask

    @data_loader
    def train_dataloader(self):
        train_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['train_set_name'], hparams, shuffle=True)
        spk_mask = self.get_mask(train_dataset, hparams['transfer_at_spkid'])

        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'], spk_mask=spk_mask)

    @data_loader
    def val_dataloader(self):
        valid_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['valid_set_name'], hparams,
                                          shuffle=False)
        spk_mask = self.get_mask(valid_dataset, hparams['transfer_at_spkid'])
        return self.build_dataloader(valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences,
                                     spk_mask=spk_mask)

    @data_loader
    def test_dataloader(self):
        test_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                         hparams['test_set_name'], hparams, shuffle=False)
        spk_mask = self.get_mask(test_dataset, [1, 2, 3, 4, 5])
        return self.build_dataloader(test_dataset, False, self.max_eval_tokens, self.max_eval_sentences,
                                     spk_mask=spk_mask)

    def original_infer(self, sample, spk_id):
        if not hparams['use_spk_id'] or spk_id == None:
            print("please use spk_id in hparams")
            exit()
        src_tokens = sample['src_tokens']
        outputs = self.original_model['model'](src_tokens, mel2ph=None, spk_embed=spk_id, f0=None, uv=None,
                                               ref_mels=None)
        return outputs

    def original_rand_spkid(self, length, new_spkid):
        new_spkid = torch.from_numpy(np.random.randint(1, new_spkid, size=length)).cuda()
        return new_spkid

    def run_model(self, model, sample, return_output=False):
        # filter the samples by spk_id
        # get origin model
        if not self.original_model_loaded and hparams['transfer_at_spkid'] != [1]:
            # checkpoint_path=hparams['original_model_path']
            # self.original_model = self.trainer.restore_model(checkpoint_path,True)
            # #self.original_model.eval()
            self.original_model = {'model': copy.deepcopy(self.model)}
            self.original_model['model'].eval()
            self.original_model_loaded = True

        hparams['global_steps'] = self.global_step
        losses = {}
        src_tokens = sample['src_tokens']  # [B, T_t]
        target = sample['targets']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        energy = sample['energy']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        output = model(src_tokens, mel2ph=mel2ph, spk_embed=spk_embed,
                       ref_mels=target, f0=f0, uv=uv, energy=energy)

        # get all outputs
        if hparams['transfer_at_spkid'] != [1]:
            original_spkid = self.original_rand_spkid(sample.get('spk_ids').shape[0], hparams['transfer_at_spkid'])
            with torch.no_grad():
                original_infer = self.original_infer(sample, original_spkid)
            output_infer = model(src_tokens,
                                 mel2ph=original_infer['mel2ph'],
                                 spk_embed=original_spkid,
                                 ref_mels=original_infer['mel_out'],
                                 f0=original_infer.get('f0'),uv=original_infer.get('uv'),
                                 energy=original_infer['energy_pred'])

        if hparams['mel_loss'] == 'l1':
            losses['mel'] = self.l1_loss(output['mel_out'], target)
            if hparams['transfer_at_spkid'] != [1]:
                losses['mel'] += self.lwf_threhold * self.l1_loss(output_infer['mel_out'], original_infer['mel_out'])
        elif hparams['mel_loss'] == 'mse':
            losses['mel'] = self.mse_loss(output['mel_out'], target)
            if hparams['transfer_at_spkid'] != [1]:
                losses['mel'] += self.lwf_threhold * self.mse_loss(output_infer['mel_out'], original_infer['mel_out'])
        else:
            losses['mel'] = torch.zeros([1]).cuda()

        losses['dur'] = self.dur_loss(output['dur'], mel2ph, src_tokens)
        if hparams['transfer_at_spkid'] != [1]:
            losses['dur'] += self.lwf_threhold * self.dur_loss(output_infer['dur'], original_infer['mel2ph'],
                                                               src_tokens)

        if hparams['use_pitch_embed']:
            p_pred = output['pitch_logits']
            losses['uv'], losses['f0'] = self.pitch_loss(p_pred, f0, uv)
            if hparams['transfer_at_spkid'] != [1]:
                losses_uv, losses_f0 = self.pitch_loss(output_infer['pitch_logits'], original_infer.get('f0'),
                                                       original_infer.get('uv'))
                losses['uv'], losses['f0'] = self.lwf_threhold * losses_uv + losses[
                    'uv'], self.lwf_threhold * losses_f0 + losses['f0']
            if losses['uv'] is None:
                del losses['uv']
        if hparams['use_energy_embed']:
            losses['energy'] = self.energy_loss(output['energy_pred'], energy)
            if hparams['transfer_at_spkid'] != [1]:
                losses['energy'] += self.lwf_threhold * self.energy_loss(output_infer['energy_pred'],
                                                                         original_infer['energy_pred'])

        if not return_output:
            return losses
        else:
            return losses, output

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
                for i in range(1, 6):
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

    def l1_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        pad_size = min(decoder_output.shape[1], target.shape[1])
        decoder_output = decoder_output[:, :pad_size, :]
        target = target[:, :pad_size, :]
        l1_loss = F.l1_loss(decoder_output, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def energy_loss(self, energy_pred, energy):
        pad_size = min(energy_pred.shape[1], energy.shape[1])
        energy_pred = energy_pred[:, :pad_size]
        energy = energy[:, :pad_size]

        nonpadding = (energy != 0).float()
        loss = (F.mse_loss(energy_pred, energy, reduction='none') * nonpadding).sum() / nonpadding.sum()
        loss = loss * hparams['lambda_energy']
        return loss


if __name__ == '__main__':
    hparams['save_gt'] = True
    Fs2lwf.start()
