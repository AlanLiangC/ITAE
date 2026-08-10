mkdir -p output/planner_eval_2hz3f_ego
for seed in 42; do
  python tools/evaluate_flow_planner.py \
    --config configs/planner/nuscenes_flow_raw_pe_2hz3f_4s.yaml \
    --manifest data/manifests/nuscenes_planner_2hz3f_front_4s/planner_final_eval.jsonl \
    --checkpoint output/planner_raw_pe_2hz3f_ego_seed${seed}/best.pt \
    --output output/planner_eval_2hz3f_ego/raw_seed${seed}

  python tools/evaluate_flow_planner.py \
    --config configs/planner/nuscenes_flow_token_v4_pe_2hz3f_4s.yaml \
    --manifest data/manifests/nuscenes_planner_2hz3f_front_4s/planner_final_eval.jsonl \
    --checkpoint output/planner_token_v4_pe_2hz3f_ego_seed${seed}/best.pt \
    --output output/planner_eval_2hz3f_ego/token_seed${seed}

  python tools/compare_flow_planners.py \
    --raw-eval output/planner_eval_2hz3f_ego/raw_seed${seed} \
    --token-eval output/planner_eval_2hz3f_ego/token_seed${seed} \
    --raw-history output/planner_raw_pe_2hz3f_ego_seed${seed}/training_history.jsonl \
    --token-history output/planner_token_v4_pe_2hz3f_ego_seed${seed}/training_history.jsonl \
    --output output/planner_eval_2hz3f_ego/comparison_seed${seed}.json
done

python tools/summarize_flow_planner_seeds.py \
  --comparisons output/planner_eval_2hz3f_ego/comparison_seed{42}.json \
  --output output/planner_eval_2hz3f_ego/three_seed_summary.json