gpu_device=1

if [ $# == 0 ]
then
    rm -r checkpoints/fs2_baseline_100_FedAVG

    CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_baseline_FedAVG.py \
    --config configs/tts/vctk/fs2_transfer.yaml \
    --exp_name "fs2_baseline_100_FedAVG" \
    --reset \
    --hparams="max_updates=50000"

    CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_baseline_FedAVG.py \
    --config configs/tts/vctk/fs2_transfer.yaml \
    --exp_name "fs2_baseline_100_FedAVG" \
    --infer \
    --hparams="save_gt=True"

else
    if [ $# == 1 -a $1 == "infer" ]
    then
        CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_baseline_FedAVG.py \
        --config configs/tts/vctk/fs2_transfer.yaml \
        --exp_name "fs2_baseline_100_FedAVG" \
        --infer \
        --hparams="save_gt=True"
    fi
fi