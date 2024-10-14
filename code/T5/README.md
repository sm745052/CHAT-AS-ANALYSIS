## Training

```sh
accelerate launch --multi_gpu --num_processes 4 code/T5/train.py --train_data data/DSTC7/train.txt --val_data data/DSTC7/val.txt --bs 12 --wandb --eval_every 7500 --model_name "t5-small"
```

change num process according to number of gpus provided.


```sh
python code/T5/infer.py --inp data/DSTC7/test_formatted.txt --out out.txt --ckpt "$folder/epoch_$epoch.pth" --bs 12
```
