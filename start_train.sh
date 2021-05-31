python run_qg.py \
    --model_name_or_path t5-base \
    --model_type t5 \
    --tokenizer_name_or_path t5_qg_tokenizer \
    --output_dir t5-base-qg-hl \
    --train_file_path data/valid_data_e2e_qg_highlight_qg_format_t5.pt \
    --valid_file_path data/train_data_e2e_qg_highlight_qg_format_t5.pt \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --seed 42 \
    --do_train \
    --do_eval \
    --logging_steps 1000