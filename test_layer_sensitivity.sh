python layer_sensitivity.py \
    --model_name_or_path gpt2 \
    --dataset_name squad \
    --pretrained_model_path /data/arpit/code/outputs/gpt2_qa_switch_precision-20251006-005433/final_model.pt \
    --use_from_pretrained \
    --original_a_bits 4 6 8 16 \
    --original_w_bits 4 6 8 16 \
    --original_ranks 16 16 16 16 \
    --original_alphas 32.0 32.0 32.0 32.0 \
    --original_dropouts 0.02 0.02 0.02 0.02 \
    --test_bits 4 6 \
    --per_device_eval_batch_size 512 \
    --max_eval_samples 5000 \
    --output_dir ./layer_sensitivity_results-20251006-005433 \
    --overwrite_output_dir
    