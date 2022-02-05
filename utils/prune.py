"""Handles all the pruning-related stuff."""
import torch
import sys
import copy


class SparsePruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, masks, args_dict, begin_prune_step, end_prune_step, now_spk_idx):
        self.model = model
        self.args_dict = args_dict
        self.sparsity_func_exponent = 3
        self.begin_prune_step = begin_prune_step
        self.end_prune_step = end_prune_step
        self.last_prune_step = begin_prune_step
        self.masks = masks

        self.now_spk_idx = now_spk_idx

        return

    def _pruning_mask(self, weights, mask, name, pruning_ratio):
        """Ranks weights by magnitude. Sets all below kth to 0.
           Returns pruned mask.
        """
        # Select all prunable weights, ie. belonging to current dataset.
        # tensor = weights[mask.eq(self.now_spk_idx) | mask.eq(0)] # This will flatten weights
        tensor = weights[mask.eq(self.now_spk_idx) | mask.eq(0)]  # This will flatten weights

        abs_tensor = tensor.abs()
        cutoff_rank = round(pruning_ratio * tensor.numel())

        try:
            cutoff_value = abs_tensor.cpu().kthvalue(cutoff_rank)[0].cuda()  # value at cutoff rank
            # print('Pruning for spk idx: %d' % self.now_spk_idx)
            # print('layer %s has been pruned by : %d/%d' % (name, cutoff_rank, tensor.numel()))
        except:
            # print('layer %s has not been pruned because the num is not enough.' % name)
            return mask
            # print("Not enough weights for pruning, that is to say, too little space for new task, need expand the network.")
            # sys.exit(2)

        # Remove those weights which are below cutoff and belong to current
        # dataset that we are training for.
        remove_mask = weights.abs().le(cutoff_value) * mask.eq(self.now_spk_idx)
        mask[remove_mask.eq(1)] = 0
        # print(mask)

        # print('Layer {}, pruned {}/{} ({:.2f}%)'.format(
        #        layer_name, mask.eq(0).sum(), tensor.numel(),
        #        float(100 * mask.eq(0).sum()) / tensor.numel()))
        return mask

    def _adjust_sparsity(self, curr_prune_step):

        if self.end_prune_step - self.begin_prune_step == 0:
            return self.args_dict['target_sparsity']

        p = min(1.0,
                max(0.0,
                    ((curr_prune_step - self.begin_prune_step)
                     / (self.end_prune_step - self.begin_prune_step))
                    ))

        sparsity = self.args_dict['target_sparsity'] + \
                   (self.args_dict['initial_sparsity'] - self.args_dict['target_sparsity']) * pow(1 - p,
                                                                                                  self.sparsity_func_exponent)

        return sparsity

    def _time_to_update_masks(self, curr_prune_step):

        if self.end_prune_step - self.begin_prune_step == 0:
            return False

        is_step_within_pruning_range = \
            (curr_prune_step >= self.begin_prune_step) and \
            (curr_prune_step <= self.end_prune_step)
        is_pruning_step = (
                                  self.last_prune_step + self.args_dict['pruning_frequency']) <= curr_prune_step

        return is_step_within_pruning_range and is_pruning_step

    def gradually_prune(self, curr_prune_step):

        if self._time_to_update_masks(curr_prune_step):
            #
            print(self.calculate_curr_task_ratio())

            self.last_prune_step = curr_prune_step
            curr_pruning_ratio = self._adjust_sparsity(curr_prune_step)

            for name, parameters in self.model.named_parameters():
                if name in self.masks:
                    mask = self._pruning_mask(parameters, self.masks[name], name, pruning_ratio=curr_pruning_ratio)
                    self.masks[name] = mask

            # print(self.masks)
            print('Pruning for spk idx: %d' % self.now_spk_idx)
            print('Pruning each layer by removing %.2f%% of values' % (100 * curr_pruning_ratio))
        else:
            curr_pruning_ratio = self._adjust_sparsity(self.last_prune_step)

        return curr_pruning_ratio

    def calculate_sparsity(self):
        total_elem = 0
        zero_elem = 0
        is_first_conv = True

        for name, parameters in self.model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                total_elem += torch.sum(mask.eq(self.now_spk_idx) | mask.eq(0))
                zero_elem += torch.sum(mask.eq(0))

            # total_elem += torch.sum(mask.ge(self.now_spk_idx) | mask.eq(0))
            # zero_elem += torch.sum(mask.eq(self.now_spk_idx))
            # break  # because every layer has the same pruning ratio,
            #        # so we are able to see only one layer for getting the sparsity
        if total_elem.cpu() != 0.0:
            return float(zero_elem.cpu()) / float(total_elem.cpu())
        else:
            return 0.0

    def calculate_curr_task_ratio(self):
        total_elem = 0
        curr_task_elem = 0
        is_first_conv = True

        for name, parameters in self.model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                total_elem += mask.numel()
                curr_task_elem += torch.sum(mask.eq(self.now_spk_idx))

        return float(curr_task_elem.cpu()) / total_elem

    def do_weight_decay_and_make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.masks
        for name, parameters in self.model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                # Set grads of all weights not belonging to current dataset to 0.
                if parameters.grad is not None:
                    parameters.grad.data[mask.ne(self.now_spk_idx)] = 0

        return

    def make_all_grad_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.masks
        for name, parameters in self.model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                # Set grads of all weights not belonging to current dataset to 0.
                # print(parameters.grad)
                if parameters.grad is not None:
                    parameters.grad.data[mask.ne(self.now_spk_idx) | mask.eq(self.now_spk_idx)] = 0
                    # parameters.grad.data[mask.ne(self.now_spk_idx)] = 0
        return

    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.masks

        for name, parameters in self.model.named_parameters():
            if name in self.masks:
                layer_mask = self.masks[name]
                parameters[layer_mask.eq(0)] = 0.0
        return

    def apply_mask(self):
        """To be done to retrieve weights just for a particular dataset."""
        for name, parameters in self.model.named_parameters():
            if name in self.masks:
                weight = parameters
                mask = self.masks[name].cuda()
                # print(self.calculate_curr_task_ratio())

                with torch.no_grad():
                    weight[mask.eq(0)] = 0.0
                    weight[mask.gt(self.now_spk_idx)] = 0.0

        return

    def make_pruned_weights_trainable(self):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        assert self.masks, 'there is no masks'

        for name, parameters in self.model.named_parameters():

            # if 'spk_embed_proj' in name or "piggymask" in name or "_copy" in name:
            #     continue
            if name in self.masks:
                mask = self.masks[name]
                mask[mask.eq(0)] = self.now_spk_idx
