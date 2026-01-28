python experiments/eval.py standard \
  --dataset_dir /path/to/lkif/datasets/save_datasets_test \
  --test_dataset test_synthetic.json \
  --encoder_spec all-MiniLM-L6-v2 \
  --llm_base_dir /path/to/lkif/Llama-3.2-1B-Instruct \
  --llm_type llama3 \
  --eval_mode kb \
  --seed 42 \
  --save_dir ./eval_results_601 \
  --exp_config_name generation_results \
  --precomputed_embed_keys_path /path/to/lkif/datasets/save_datasets_test/test_synthetic_all-MiniLM-L6-v2_embd_key.npy \
  --precomputed_embed_values_path /path/to/lkif/datasets/save_datasets_test/test_synthetic_all-MiniLM-L6-v2_embd_value.npy \
  --kb_layer_frequency 3 \
  --subset_size 100 \
  --sample_size 5 \
  --topk_size -1

