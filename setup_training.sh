
pip install -r requirements.txt

mkdir ./pretrained_ckpt
mkdir ./datasets
mkdir ./datasets/COCO17

wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -O ./pretrained_ckpt/yolox_s.pth
wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth -O ./pretrained_ckpt/yolox_l.pth


wget -c https://storage.googleapis.com/coco-mintrain/annotations_trainval2017.zip
wget -c https://storage.googleapis.com/coco-mintrain/minitrain2017.zip
wget -c https://storage.googleapis.com/coco-mintrain/val2017.zip
wget -c https://storage.googleapis.com/coco-mintrain/test2017.zip
wget -c instances_minitrain2017.json

unzip minitrain2017.zip -d ./datasets/COCO17
unzip val2017.zip -d ./datasets/COCO17
unzip test2017.zip -d ./datasets/COCO17
unzip annotations_trainval2017.zip -d ./datasets/COCO17
mv instances_minitrain2017.json ./datasets/COCO17/annotations/

rm minitrain2017.zip
rm val2017.zip
rm test2017.zip
rm annotations_trainval2017.zip

