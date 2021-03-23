#!/usr/bin/env bash
python run_common_voice.py \
    --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
    --dataset_config_name="nl" \
    --output_dir=/workspace/output_models/tr/wav2vec2-large-xlsr-dutch-demo \
    --overwrite_output_dir \
    --num_train_epochs="30" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --evaluation_strategy="steps" \
    --learning_rate="3e-4" \
    --warmup_steps="500" \
    --fp16 \
    --freeze_feature_extractor \
    --save_steps="400" \
    --eval_steps="400" \
    --save_total_limit="2" \
    --logging_steps="100" \
    --group_by_length \
    --feat_proj_dropout="0.0" \
    --layerdrop="0.1" \
    --gradient_checkpointing \
    --do_train --do_eval \
    --preprocessing_num_workers="4"