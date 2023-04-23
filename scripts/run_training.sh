export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="runs/dog-cat-pose-{timestamp}"
export HUB_MODEL_ID="dog-cat-pose"
export DISK_DIR="/mnt/disks/persist/" 
export DATASET_DIR="/mnt/disks/persist/dataset"

python3 train_controlnet_flax.py  --pretrained_model_name_or_path=$MODEL_DIR  --output_dir=$OUTPUT_DIR  \
--train_data_dir=$DATASET_DIR  --resolution=512  --learning_rate=1e-5  --train_batch_size=2  --revision="non-ema" \
 --from_pt  0 --dataloader_num_workers=16   --validation_image "./cond1.jpg" "./cond2.jpg"  \
--validation_prompt "a tortoiseshell cat is sitting on a cushion" "a yellow dog standing on a lawn"  \
--validation_steps=1000  --train_batch_size=2  --revision="non-ema"  --from_pt  --report_to="wandb"  \
--tracker_project_name=$HUB_MODEL_ID  --num_train_epochs=50 --image_column "original_image" --caption_column "caption"  --cache_dir=$DISK_DIR
