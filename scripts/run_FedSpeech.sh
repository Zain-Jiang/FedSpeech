sparsity_array=(1.0 1.0 1.0 0.7 0.7 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.0)
task_num=10
gpu_device=1

if [ $1 == "first_round" ]; then
	echo "start first_round"
    rm -r checkpoints/FedSpeech*
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

        CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_FedSpeech.py \
            --config configs/tts/vctk/fs2_transfer.yaml \
            --exp_name FedSpeech \
            --reset \
            --hparams="max_updates=$((max_updates)),mode=prune,hidden_size=$((256 + expanded_size * 32)),record_name=FedSpeech_record,predictor_hidden=$((256 + expanded_size * 32)),lr=3"
        max_updates=$((max_updates + 1000))

        CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_FedSpeech.py \
            --config configs/tts/vctk/fs2_transfer.yaml \
            --exp_name FedSpeech \
            --reset \
            --hparams="max_updates=$((max_updates)),mode=prune,hidden_size=$((256 + expanded_size * 32)),record_name=FedSpeech_record,predictor_hidden=$((256 + expanded_size * 32)),pruning_step_num=0,lr=2"
        max_updates=$((max_updates + 3000))
    done

    mkdir checkpoints/FedSpeech_copy
    cp -r checkpoints/FedSpeech/* checkpoints/FedSpeech_copy
fi


if [ $1 == "second_round" ]; then
    echo "start second_round"
    spk_ids=(3 4 5 6 7 8 9 10 11 12)
#    spk_ids=(5)
    max_updates=41000
	expanded_size=0

    rm -r checkpoints/FedSpeech/generated*

    for i in ${spk_ids[@]}; do
#        expanded_size=$[$i/2]
        expanded_size=6
        sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
        sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
        sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
        sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
        echo "pruning_step_num: 3000" >>configs/tts/vctk/fs2_transfer.yaml
        echo "initial_sparsity: 0.0" >>configs/tts/vctk/fs2_transfer.yaml
        echo "target_sparsity: ${sparsity_array[i]}" >>configs/tts/vctk/fs2_transfer.yaml
        echo "transfer_at_spkid: [$i]" >>configs/tts/vctk/fs2_transfer.yaml
        rm -f checkpoints/FedSpeech/model_ckpt_s*
        cp -r checkpoints/FedSpeech_copy/model_ckpt_steps_40000.ckpt checkpoints/FedSpeech/
        CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_FedSpeech.py \
            --config configs/tts/vctk/fs2_transfer.yaml \
            --exp_name FedSpeech \
            --reset \
            --hparams="max_updates=$((max_updates)),mode=finetune,selective=new,val_check_interval=100,record_name=FedSpeech_record,hidden_size=$((256 + expanded_size * 32)),predictor_hidden=$((256 + expanded_size * 32)),lr=0.5"

        CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_FedSpeech.py --config configs/tts/vctk/fs2_transfer.yaml \
            --exp_name FedSpeech --infer --hparams="save_gt=True,mode=infer,selective=new,hidden_size=$((256 + expanded_size * 32)),predictor_hidden=$((256 + expanded_size * 32))"

    done
fi
