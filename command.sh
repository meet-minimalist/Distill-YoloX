# python tools/train_distill.py --student-name yolox-s_distill --teacher-name yolox-l_distill --teacher-ckpt ./pretrained_ckpt/yolox_l.pth -d 0 -b 12 -o --fp16

# training with train set and drive checkpointing
python tools/train_distill.py --student-name yolox-s_distill --teacher-name yolox-l_distill --teacher-ckpt "/content/drive/MyDrive/ThesisData/YoloXCheckpoints/yolox_l_normal/best_ckpt.pth" -d 1 -b 12 -o --fp16 data_dir "./datasets/COCO17/" train_ann "instances_minitrain2017.json" max_epoch 60 output_dir "/content/drive/MyDrive/ThesisData/YoloXCheckpoints/yolox_s_distill/"

python tools/train_distill.py --student-name yolox-s_voc_distill --teacher-name yolox-l_voc_distill --teacher-ckpt "./YOLOX_outputs/VOC0712/yolox_l_voc_colab/best_ckpt.pth" -d 1 -b 1 -o --fp16 data_dir "M:/Datasets/PASCAL/" max_epoch 60 

# training with valid set and drive checkpointing
# python tools/train_distill.py --student-name yolox-s_distill --teacher-name yolox-l_distill --teacher-ckpt ./pretrained_ckpt/yolox_l.pth -d 0 -b 12 -o --fp16 data_dir "./datasets/COCO17/" train_ann "instances_val2017.json" max_epoch 100 output_dir "/content/drive/MyDrive/ThesisData/YoloXCheckpoints/yolox_s_distill/"

# Train on minitrain-2017 Normal training
# Yolox-S
python tools/train.py -n yolox-s -d 1 -b 32 --fp16 -o data_dir "./datasets/COCO17/" train_ann "instances_minitrain2017.json" max_epoch 60 output_dir "/content/drive/MyDrive/ThesisData/YoloXCheckpoints/"

# Yolox-L
python tools/train.py -n yolox-l -d 1 -b 16 --fp16 -o data_dir "./datasets/COCO17/" train_ann "instances_minitrain2017.json" max_epoch 60 output_dir "/content/drive/MyDrive/ThesisData/YoloXCheckpoints/"

# Resume training
python tools/train.py -n yolox-l -d 1 -b 16 --fp16 -o --resume --ckpt "/content/drive/MyDrive/ThesisData/YoloXCheckpoints/yolox_s_normal/yolox_l/latest_ckpt.pth" data_dir "./datasets/COCO17/" train_ann "instances_minitrain2017.json" max_epoch 60 output_dir "/content/drive/MyDrive/ThesisData/YoloXCheckpoints/" 

