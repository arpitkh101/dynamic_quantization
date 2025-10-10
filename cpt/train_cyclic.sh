source /data/arpit/virtualenv/venv101/bin/activate

python3 train_cyclic.py \
    --is_cyclic_precision \
    --cyclic_a_bits_schedule 4 8 \
    --cyclic_w_bits_schedule 4 8 \
    --num_cyclic_period 32 \
    --do_train \
    --do_eval \
    --max_steps 1000 \
    --eval_steps 510 \
    --save_steps 1000 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 256 \
    --learning_rate 1e-3 \
    --ranks 16 \
    --alphas 32.0 \
    --dropouts 0.01 \