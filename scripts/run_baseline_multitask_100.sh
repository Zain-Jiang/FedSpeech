gpu_device=2

if [ $# == 0 ]
then
    rm -r checkpoints/fs2_multitask_100
    CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_multitask_100.py \
    --config configs/tts/vctk/fs2_transfer.yaml \
    --exp_name fs2_multitask_100 \
    --reset \
    --hparams="max_updates=50000,hidden_size=448"

    CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_multitask_100.py \
    --config configs/tts/vctk/fs2_transfer.yaml \
    --exp_name fs2_multitask_100 \
    --infer \
    --hparams="save_gt=True,hidden_size=448"
else
    if [ $# == 1 -a $1 == "infer" ]
    then
        CUDA_VISIBLE_DEVICES=$gpu_device python tasks/fs2_multitask_100.py \
        --config configs/tts/vctk/fs2_transfer.yaml \
        --exp_name fs2_multitask_100 \
        --infer \
        --hparams="save_gt=True"
    fi
fi
