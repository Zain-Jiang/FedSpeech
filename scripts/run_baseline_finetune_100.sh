gpu_device=2
spk_ids=(3)
name='finetune'

if [ $# == 0 ]
then
    rm -r checkpoints/fs2_baseline_finetune_100_*
    for spk_id in ${spk_ids[@]}
    do
        CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_finetune_100.py \
        --config configs/tts/vctk/fs2_transfer.yaml \
        --exp_name "fs2_baseline_finetune_100_$((spk_id))" \
        --reset \
        --hparams="spk_id=[$((spk_id))],max_updates=5000"

        CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_finetune_100.py \
        --config configs/tts/vctk/fs2_transfer.yaml \
        --exp_name "fs2_baseline_finetune_100_$((spk_id))" \
        --infer \
        --hparams="save_gt=True,spk_id=[$((spk_id))]"
    done

    mkdir checkpoints/fs2_baseline_finetune_100_3_copy
    cp -r checkpoints/fs2_baseline_finetune_100_3/* checkpoints/fs2_baseline_finetune_100_3_copy/

    spk_ids=(4 5 6 7 8 9 10 11 12)
    for spk_id in ${spk_ids[@]}
    do
        mkdir checkpoints/fs2_baseline_finetune_100_$((spk_id))
        cp -r checkpoints/fs2_baseline_finetune_100_3_copy/* checkpoints/fs2_baseline_finetune_100_$((spk_id))/

        CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_finetune_100.py \
        --config configs/tts/vctk/fs2_transfer.yaml \
        --exp_name "fs2_baseline_finetune_100_$((spk_id))" \
        --reset \
        --hparams="spk_id=[$((spk_id))],max_updates=10000"

        CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_finetune_100.py \
        --config configs/tts/vctk/fs2_transfer.yaml \
        --exp_name "fs2_baseline_finetune_100_$((spk_id))" \
        --infer \
        --hparams="save_gt=True,spk_id=[$((spk_id))]"
    done

else
    if [ $# == 1 -a $1 == "infer" ]
    then
        for spk_id in ${spk_ids[@]}
        do
            CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_finetune_100.py \
            --config configs/tts/vctk/fs2_transfer.yaml \
            --exp_name "fs2_baseline_finetune_100_$((spk_id))" \
            --infer \
            --hparams="save_gt=True,spk_id=[$((spk_id))]"
        done
    fi
fi