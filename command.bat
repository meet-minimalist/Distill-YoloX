 
@REM train on validation set
 python tools/train_distill.py --student-name yolox-s_distill --teacher-name yolox-l_distill --teacher-ckpt ./pretrained_ckpt/yolox_l.pth -d 0 -b 1 -o --fp16 data_dir "M:/Datasets/COCO17/" train_ann "instances_val2017.json"
@REM train on train set
 python tools/train_distill.py --student-name yolox-s_distill --teacher-name yolox-l_distill --teacher-ckpt ./pretrained_ckpt/yolox_l.pth -d 0 -b 1 -o --fp16 data_dir "M:/Datasets/COCO17/" train_ann "instances_train2017.json" max_epoch 100 output_dir "./YOLOX_outputs/"

@REM train on train set with drive checkpointing
 python tools/train_distill.py --student-name yolox-s_distill --teacher-name yolox-l_distill --teacher-ckpt ./pretrained_ckpt/yolox_l.pth -d 0 -b 1 -o --fp16 data_dir "M:/Datasets/COCO17/" train_ann "instances_train2017.json" max_epoch 100 output_dir "/content/drive/MyDrive/ThesisData/YoloXCheckpoints/yolox_s_distill/"

@REM  python tools/train_distill.py --student-name yolox-s_distill --teacher-name yolox-l_distill --teacher-ckpt ./pretrained_ckpt/yolox_l.pth -d 0 -b 2 --fp16 -o