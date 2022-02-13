
pip install -r requirements.txt

mkdir ./pretrained_ckpt
mkdir ./datasets
mkdir ./datasets/COCO17

wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -O ./pretrained_ckpt/yolox_s.pth
wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth -O ./pretrained_ckpt/yolox_l.pth


wget -c https://storage.googleapis.com/coco-mintrain/annotations_trainval2017.zip
wget -c https://storage.googleapis.com/coco-mintrain/instances_minitrain2017.json
unzip annotations_trainval2017.zip -d ./datasets/COCO17
mv instances_minitrain2017.json ./datasets/COCO17/annotations/
rm annotations_trainval2017.zip

wget -c https://storage.googleapis.com/coco-mintrain/minitrain2017.zip
unzip minitrain2017.zip -d ./datasets/COCO17
rm minitrain2017.zip

wget -c https://storage.googleapis.com/coco-mintrain/val2017.zip
unzip val2017.zip -d ./datasets/COCO17
rm val2017.zip

wget -c https://storage.googleapis.com/coco-mintrain/test2017.zip
unzip test2017.zip -d ./datasets/COCO17
rm test2017.zip

