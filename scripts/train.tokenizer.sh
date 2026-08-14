export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  --module tools.training.train_tokenizer \
  --config configs/tokenizer/navsim_trainval_v4_scratch_4s.yaml \
  --sources navsim \
  --output output/navsim_trainval_v4_scratch_4gpu \
  --no-resume