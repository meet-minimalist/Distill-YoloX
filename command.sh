# python tools/train_distill.py --student-name yolox-s_distill --teacher-name yolox-l_distill --teacher-ckpt ./pretrained_ckpt/yolox_l.pth -d 0 -b 12 -o --fp16

# training with train set and drive checkpointing
python tools/train_distill.py --student-name yolox-s_distill --teacher-name yolox-l_distill --teacher-ckpt ./pretrained_ckpt/yolox_l.pth -d 1 -b 12 -o --fp16 data_dir "./datasets/COCO17/" train_ann "instances_train2017.json" max_epoch 100 output_dir "/content/drive/MyDrive/ThesisData/YoloXCheckpoints/yolox_s_distill/"

# training with valid set and drive checkpointing
# python tools/train_distill.py --student-name yolox-s_distill --teacher-name yolox-l_distill --teacher-ckpt ./pretrained_ckpt/yolox_l.pth -d 0 -b 12 -o --fp16 data_dir "./datasets/COCO17/" train_ann "instances_val2017.json" max_epoch 100 output_dir "/content/drive/MyDrive/ThesisData/YoloXCheckpoints/yolox_s_distill/"

# Train on minitrain-2017 Normal training
python tools/train.py -n yolox-s -d 1 -b 32 --fp16 -o --fp16 data_dir "./datasets/COCO17/" train_ann "instances_minitrain2017.json" max_epoch 60 output_dir "/content/drive/MyDrive/ThesisData/YoloXCheckpoints/yolox_s_normal/"

