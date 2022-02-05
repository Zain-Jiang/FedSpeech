import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import shutil
import os
import sys
import copy
from multiprocessing.pool import Pool
from tqdm import tqdm
import pickle
import collections

from utils.pl_utils import data_loader
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset
from utils.plot import plot_to_figure
from utils.world_utils import restore_f0, process_f0
from utils import audio
from utils.prune import SparsePruner
import utils

import numpy as np

from modules.tts_modules import DurationPredictorLoss
from modules.fs2 import FastSpeech2
from tasks.fs2 import FastSpeechDataset, FastSpeech2Task
from tasks.transformer_tts import TransformerTtsTask
from tasks.base_task import BaseDataset

import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


class selectiveMask_Apply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model, selectiveMask_flatten, threshold):
        index = 0
        for name, parameters in model.named_parameters():
            mask_thresholded = Binarizer.apply(
                selectiveMask_flatten[index:index + torch.flatten(parameters).size()[0]].view(parameters.size()),
                threshold)
            parameters = mask_thresholded * parameters
            index += torch.flatten(parameters).size()[0]
        return model

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


class Fs2FedSpeechTask(FastSpeech2Task):
    def __init__(self):
        super(Fs2FedSpeechTask, self).__init__()
        self.dur_loss_fn = DurationPredictorLoss()
        self.mse_loss_fn = torch.nn.MSELoss()
        self.pruning_mask = None
        self.pruner = None
        self.DEFAULT_THRESHOLD = 5e-3
        self.selectiveMask_flatten = None
        self.threshold_fn = Binarizer.apply

    def get_spk_mask(self, dataset, spk_ids):
        mask = torch.zeros(len(dataset))
        for index, item in enumerate(dataset):
            if item['spk_id'] in spk_ids:
                mask[index] = 1
        return mask

    def get_pruning_mask(self, model):
        masks = {}
        for name, parameters in model.named_parameters():
            if 'spk_embed_proj' in name or "selectiveMask" in name:
                continue

            # print(name,':',parameters.size())
            mask = torch.ByteTensor(parameters.size()).fill_(0)
            # if 'cuda' in parameters.type():
            mask = mask.cuda()
            masks[name] = mask
        return masks

    def expand_mask(self, masks, previous_masks, model):
        # load the state_dict on the model automatically
        for name, param in model.named_parameters():
            if name in masks:
                mask_param = previous_masks[name]
                if len(masks[name].size()) == 1:
                    if masks[name].size(0) < mask_param.size(0):
                        masks[name].copy_(mask_param[:masks[name].size(0)])
                    else:
                        masks[name][:mask_param.size(0)].copy_(mask_param)
                elif len(masks[name].size()) == 2:
                    if masks[name].size(0) < mask_param.size(0) or masks[name].size(1)<mask_param.size(1):
                        masks[name].copy_(mask_param[:masks[name].size(0), :masks[name].size(1)])
                    else:
                        masks[name][:mask_param.size(0), :mask_param.size(1)].copy_(mask_param)
                elif len(masks[name].size()) == 3:
                    if masks[name].size(0) < mask_param.size(0) or masks[name].size(1)<mask_param.size(1) or masks[name].size(2)<mask_param.size(2):
                        masks[name].copy_(mask_param[:masks[name].size(0), :masks[name].size(1), :masks[name].size(2)])
                    else:
                        masks[name][:mask_param.size(0), :mask_param.size(1), :mask_param.size(2)].copy_(mask_param)
                else:
                    try:
                        masks[name].copy_(mask_param)
                    except:
                        print("There is some corner case that we haven't tackled when load_state_dict.")
        return masks

    def set_selectiveMask(self, model):
        print("set_selective")
        flatten_size = 0
        for name, parameters in model.named_parameters():
            if 'spk_embed_proj' in name:
                continue
            flatten_size += torch.flatten(parameters).size()[0]
        self.selectiveMask_flatten = Parameter(torch.Tensor(flatten_size).fill_(0.01), requires_grad=True)


    def apply_selective_new(self):
        self.pruner.model = self.model
        index = 0
        for name, parameters in self.model.named_parameters():
            if 'spk_embed_proj' in name or "norm" in name:
                continue

            # parameters.detach_()
            shaped_selectiveMask = self.selectiveMask_flatten[
                               index:index + torch.flatten(parameters.detach()).size()[0]].view(
                parameters.detach().size())
            selectiveMask_thresholded = self.threshold_fn(
                shaped_selectiveMask,
                self.DEFAULT_THRESHOLD)
            with torch.no_grad():
                pruning_spkID_idx = torch.where(self.pruner.masks[name] == hparams['transfer_at_spkid'][0])
                selectiveMask_thresholded[pruning_spkID_idx] = 1.0
            parameters *= selectiveMask_thresholded
            index += torch.flatten(parameters.detach()).size()[0]


    @data_loader
    def train_dataloader(self):
        train_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['train_set_name'], hparams, shuffle=True)
        print(hparams['transfer_at_spkid'])
        spk_mask = self.get_spk_mask(train_dataset, hparams['transfer_at_spkid'])
        idx_mask = np.where(spk_mask == 1)[0]
        rand_mask = np.random.choice(idx_mask, len(idx_mask) - 100, replace=False)
        spk_mask[rand_mask] = 0
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'], spk_mask=spk_mask)

    @data_loader
    def val_dataloader(self):
        valid_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['valid_set_name'], hparams,
                                          shuffle=False)
        spk_mask = self.get_spk_mask(valid_dataset, hparams['transfer_at_spkid'])
        return self.build_dataloader(valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences,
                                     spk_mask=spk_mask)

    @data_loader
    def test_dataloader(self):
        test_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                         hparams['test_set_name'], hparams, shuffle=False)
        spk_mask = self.get_spk_mask(test_dataset, hparams['transfer_at_spkid'])
        return self.build_dataloader(test_dataset, False, self.max_eval_tokens, self.max_eval_sentences,
                                     spk_mask=spk_mask)

    def build_model(self):
        arch = self.arch
        model = FastSpeech2(arch, self.phone_encoder)

        # load the mask
        if os.path.exists('./checkpoints/FedSpeech/pruning_mask.pkl'):
            print("Find an existed pruning mask, loading.")
            previous_pruning_mask = pickle.load(open('./checkpoints/FedSpeech/pruning_mask.pkl', 'rb'))
            self.pruning_mask = self.expand_mask(self.get_pruning_mask(model), previous_pruning_mask, model)
        else:
            print("There is no pruning mask, create a new one.")
            self.pruning_mask = self.get_pruning_mask(model)

        # construct the pruner
        args_dict = {'mode': 'prune',
                     'pruning_frequency': 100,
                     'weight_decay': 0,
                     'initial_sparsity': hparams['initial_sparsity'],
                     'target_sparsity': hparams['target_sparsity'], }
        self.pruner = SparsePruner(model, self.pruning_mask, args_dict,
                                   hparams['max_updates'] - hparams['pruning_step_num'],
                                   hparams['max_updates'],
                                   hparams['transfer_at_spkid'][0])

        # set the previous pruned mask to spk_idx to make it trainable
        if hparams['mode'] == "prune" and hparams['pruning_step_num'] != 0:
            self.pruner.make_pruned_weights_trainable()

        # set selectiveMask and make requires_grad false
        if hparams['mode'] == "finetune":
            self.set_selectiveMask(model)
            for name, parameters in model.named_parameters():
                if 'spk_embed_proj' in name or "norm" in name:
                    parameters.requires_grad = True
                    continue
                parameters.requires_grad = False
            #save the initialized mask
            if hparams.get('selective', -1) == "new":
                print("save selective_mask_new")
                pickle.dump(self.selectiveMask_flatten,
                            open('./checkpoints/FedSpeech/selective_mask_new' + str(
                                hparams['transfer_at_spkid'][0]) + '.pkl',
                                 'wb'))

        if hparams['mode'] == "freezing":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.spk_embed_proj.parameters():
                param.requires_grad = True
        # get selective mask when infer

        if hparams.get('selective', -1) == "new" and hparams['mode'] == "infer":
            print("load_selective_new")
            self.selectiveMask_flatten = pickle.load(
                open('./checkpoints/FedSpeech/selective_mask_new' + str(
                    hparams['transfer_at_spkid'][0]) + '.pkl',
                     'rb'))

        return model

    def build_optimizer(self, model):

        self.optimizer = optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=4e-5)
        if hparams['mode'] == "finetune":
            self.optimizer.add_param_group({'params': self.selectiveMask_flatten})

        return optimizer

    def on_save_checkpoint(self, checkpoint):
        if hparams['mode'] == "prune":
            print("save pruning_mask")
            pickle.dump(self.pruning_mask, open('./checkpoints/FedSpeech/pruning_mask.pkl', 'wb'))

        if hparams['mode'] == "finetune":
            self.model = copy.deepcopy(self.model_copy)
            if hparams.get('selective', -1) == "new":
                print("save selective_mask_new")
                pickle.dump(self.selectiveMask_flatten,
                            open('./checkpoints/FedSpeech/selective_mask_new' + str(
                                hparams['transfer_at_spkid'][0]) + '.pkl',
                                 'wb'))

    def on_sanity_check_start(self):
        # because of expansion, clearing the optimizer's state_dict
        self.optimizer.state = collections.defaultdict(dict)

        self.model_copy = copy.deepcopy(self.model)
        if hparams['mode'] == "finetune" and (hparams.get('selective', -1) == -1 or hparams.get('selective', -1) == ""):
            self.pruner.apply_mask()

    def _training_step(self, sample, batch_idx, _):

        if hparams['mode'] == "finetune" and hparams.get('selective', -1) == "new":
            self.apply_selective_new()

        loss_output = self.run_model(self.model, sample)
        total_loss = sum([v for v in loss_output.values() if v.requires_grad])
        loss_output['batch_size'] = sample['targets'].size()[0]

        return total_loss, loss_output

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        # Set fixed param grads to 0.
        if hparams['mode'] == "prune":
            self.pruner.do_weight_decay_and_make_grads_zero()
        elif hparams['mode'] == "finetune":
            self.pruner.make_all_grad_zero()
        elif hparams['mode'] == "freezing":
            self.pruner.make_all_grad_zero()
        else:
            print('please choose a mode in hparams')
            sys.exit(0)

        optimizer.step()

        # Set pruned weights to 0.
        if hparams['mode'] == "prune":
            curr_pruning_ratio = self.pruner.gradually_prune(self.global_step)
            if self.global_step >= hparams['max_updates'] - hparams['pruning_step_num']:
                self.pruner.apply_mask()

        optimizer.zero_grad()
        self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

        if hparams['mode'] == "finetune":
            del self.model
            self.model = copy.deepcopy(self.model_copy)

    def validation_step(self, sample, batch_idx):

        if hparams['mode'] == "finetune" and hparams.get('selective', -1) == "new":
            self.apply_selective_new()

        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True)
        outputs['total_loss'] = outputs['losses']['mel']
        outputs['nmels'] = sample['nmels']
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < 10:
            if 'pitch_logits' in model_out:
                f0 = sample['f0']
                uv = sample['uv']
                f0[uv > 0] = -2
                pitch_pred = model_out['pitch_logits'][:, :, 0]
                pitch_pred[model_out['pitch_logits'][:, :, 1] > 0] = -2
                self.logger.experiment.add_figure(f'pitch_{batch_idx}', plot_to_figure({
                    'gt': f0[0].detach().cpu().numpy(),
                    'pred': pitch_pred[0].detach().cpu().numpy()
                }), self.global_step)

        if hparams['mode'] == "finetune":
            del self.model
            self.model = copy.deepcopy(self.model_copy)

        return outputs

    def validation_end(self, outputs):
        loss_output = self._validation_end(outputs)
        print(f"\n==============\n "
              f"valid results: {loss_output}"
              f"\n==============\n")
        with open("valid_loss_FedSpeech_"+str(hparams['record_name'])+".txt", 'a+') as file_object:
            file_object.write(f"valid results: {loss_output}\n")
        return {
            'log': {f'val/{k}': v for k, v in loss_output.items()},
            'val_loss': loss_output['total_loss']
        }

    def test_step(self, sample, batch_idx):
        """Performs evaluation."""

        if hparams['mode'] == "infer" and hparams.get('selective', -1) == "new":
            self.apply_selective_new()

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
                for i in range(1, 16):
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
    Fs2FedSpeechTask.start()
