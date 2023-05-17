NUM_TRAINERS=$(python -c "import torch;print(torch.cuda.device_count())")

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=$NUM_TRAINERS "$1" --config-file "$2"