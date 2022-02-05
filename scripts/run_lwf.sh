#lwf first_step
#sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
#sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
#echo "transfer_at_spkid: [1]" >> configs/tts/vctk/fs2_transfer.yaml
# CUDA_VISIBLE_DEVICES=1 python tasks/fs2_transfer_lwf.py --config configs/tts/vctk/fs2_transfer.yaml \
# --exp_name fs2_lwf_spk1 --reset --hparams="transfer_at_spkid=[1]" --hparams="max_updates=20000"

#lwf next_steps
sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
echo "transfer_at_spkid: [2]" >> configs/tts/vctk/fs2_transfer.yaml
CUDA_VISIBLE_DEVICES=1 python tasks/fs2_transfer_lwf.py --config configs/tts/vctk/fs2_transfer.yaml \
--exp_name fs2_lwf_spk1 --reset --hparams="max_updates=24000" \

sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
echo "transfer_at_spkid: [3]" >> configs/tts/vctk/fs2_transfer.yaml
CUDA_VISIBLE_DEVICES=1 python tasks/fs2_transfer_lwf.py --config configs/tts/vctk/fs2_transfer.yaml \
--exp_name fs2_lwf_spk1 --reset --hparams="max_updates=28000" \

sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
echo "transfer_at_spkid: [4]" >> configs/tts/vctk/fs2_transfer.yaml
CUDA_VISIBLE_DEVICES=1 python tasks/fs2_transfer_lwf.py --config configs/tts/vctk/fs2_transfer.yaml \
--exp_name fs2_lwf_spk1 --reset  --hparams="max_updates=32000" \

sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
sed -i '$d' configs/tts/vctk/fs2_transfer.yaml
echo "transfer_at_spkid: [5]" >> configs/tts/vctk/fs2_transfer.yaml
CUDA_VISIBLE_DEVICES=1 python tasks/fs2_transfer_lwf.py --config configs/tts/vctk/fs2_transfer.yaml \
--exp_name fs2_lwf_spk1 --reset --hparams="max_updates=36000" \

#infer
CUDA_VISIBLE_DEVICES=1 python tasks/fs2_transfer_lwf.py --config configs/tts/vctk/fs2_transfer.yaml \
--exp_name fs2_lwf_spk1 --infer --hparams="save_gt=True"

#cp -f checkpoints/fs2_lwf_spk1_copy2/model_ckpt_steps_20000.ckpt checkpoints/fs2_lwf_spk1/model_ckpt_steps_20000.ckpt
