python -m tools.features.cache_vggt_omega_features \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml \
  --manifest data/manifests/navsim_mini_train_4s.jsonl \
  --output /home/alan/AlanLiang/Dataset/vggt_omega_cache/navsim_mini_front_4s_train_rich \
  --batch-size 1 --shard-size 128 --num-workers 2

python -m tools.features.cache_vggt_omega_features \
  --config configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml \
  --manifest data/manifests/navsim_mini_val_4s.jsonl \
  --output /home/alan/AlanLiang/Dataset/vggt_omega_cache/navsim_mini_front_4s_val_rich \
  --batch-size 1 --shard-size 128 --num-workers 2