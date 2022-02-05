sparsity_array=(1.0 1.0 1.0 0.7 0.7 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.0)
task_num=10
gpu_device=0

rm -r checkpoints/fs2_CPG*
spk_ids=(3 4 5 6 7 8 9 10 11 12)
max_updates=3000
expanded_size=0
for i in ${spk_ids[@]}; do
	echo $((i))
	expanded_size=$[$i/2]
#        expanded_size=6
	sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
	sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
	sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
	sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
	echo "pruning_step_num: 2000" >>configs/tts/vctk/fs2_transfer.yaml
	echo "initial_sparsity: 0.0" >>configs/tts/vctk/fs2_transfer.yaml
	echo "target_sparsity: ${sparsity_array[i]}" >>configs/tts/vctk/fs2_transfer.yaml
	echo "transfer_at_spkid: [$i]" >>configs/tts/vctk/fs2_transfer.yaml

	CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_CPG.py \
		--config configs/tts/vctk/fs2_transfer.yaml \
		--exp_name fs2_CPG \
		--reset \
		--hparams="max_updates=$((max_updates)),mode=prune,hidden_size=$((256 + expanded_size * 32)),record_name=expand_old,predictor_hidden=$((256 + expanded_size * 32)),lr=3"
	max_updates=$((max_updates + 1000))

	CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_CPG.py \
		--config configs/tts/vctk/fs2_transfer.yaml \
		--exp_name fs2_CPG \
		--reset \
		--hparams="max_updates=$((max_updates)),mode=prune,hidden_size=$((256 + expanded_size * 32)),record_name=expand_old,predictor_hidden=$((256 + expanded_size * 32)),pruning_step_num=0,lr=2"
	max_updates=$((max_updates + 3000))
done

mkdir checkpoints/fs2_CPG_copy
cp -r checkpoints/fs2_CPG/* checkpoints/fs2_CPG_copy


echo "start old_piggy"
spk_ids=(3 4 5 6 7 8 9 10 11 12)
#    spk_ids=(12)
max_updates=41000
expanded_size=0

rm -r checkpoints/fs2_CPG/generated*

for i in ${spk_ids[@]}; do
	expanded_size=$[$i/2]
	sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
	sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
	sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
	sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
	echo "pruning_step_num: 3000" >>configs/tts/vctk/fs2_transfer.yaml
	echo "initial_sparsity: 0.0" >>configs/tts/vctk/fs2_transfer.yaml
	echo "target_sparsity: ${sparsity_array[i]}" >>configs/tts/vctk/fs2_transfer.yaml
	echo "transfer_at_spkid: [$i]" >>configs/tts/vctk/fs2_transfer.yaml
	rm -f checkpoints/fs2_CPG/model_ckpt_s*
	cp -r checkpoints/fs2_CPG_copy/model_ckpt_steps_40000.ckpt checkpoints/fs2_CPG/

	CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_CPG.py \
		--config configs/tts/vctk/fs2_transfer.yaml \
		--exp_name fs2_CPG \
		--reset \
		--hparams="max_updates=$((max_updates)),mode=finetune,val_check_interval=100,record_name=expand_piggyold,hidden_size=$((256 + expanded_size * 32)),predictor_hidden=$((256 + expanded_size * 32)),lr=0.5"

	CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_CPG.py \
		--config configs/tts/vctk/fs2_transfer.yaml \
		--exp_name fs2_CPG \
		--infer \
		--hparams="save_gt=True,mode=infer,piggy=old,hidden_size=$((256 + expanded_size * 32)),predictor_hidden=$((256 + expanded_size * 32))"
done