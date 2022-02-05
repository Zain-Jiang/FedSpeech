gpu_device=1
spk_ids=(3 4 5 6 7 8 9 10 11 12)
#spk_ids=(6)


if [ $# == 0 ]
then
    rm -r checkpoints/fs2_baseline_100_*
    for spk_id in ${spk_ids[@]}
    do
        CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_scratch_100.py \
        --config configs/tts/vctk/fs2_transfer.yaml \
        --exp_name "fs2_baseline_100_$((spk_id))" \
        --reset \
        --hparams="spk_id=[$((spk_id))],max_updates=5000"

        CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_scratch_100.py \
        --config configs/tts/vctk/fs2_transfer.yaml \
        --exp_name "fs2_baseline_100_$((spk_id))" \
        --infer \
        --hparams="save_gt=True"
    done

else
    if [ $# == 1 -a $1 == "infer" ]
    then
        for spk_id in ${spk_ids[@]}
        do
            CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_scratch_100.py \
            --config configs/tts/vctk/fs2_transfer.yaml \
            --exp_name "fs2_baseline_100_$((spk_id))" \
            --infer \
            --hparams="save_gt=True"
        done
    fi
fi