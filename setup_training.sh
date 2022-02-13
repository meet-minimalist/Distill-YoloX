
pip install -r requirements.txt

mkdir ./pretrained_ckpt

wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -O ./pretrained_ckpt/yolox_s.pth
wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth -O ./pretrained_ckpt/yolox_m.pth
wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth -O ./pretrained_ckpt/yolox_l.pth
wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth -O ./pretrained_ckpt/yolox_x.pth

wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth -O ./pretrained_ckpt/yolox_nano.pth
wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth -O ./pretrained_ckpt/yolox_tiny.pth

mkdir ./datasets
cd ./datasets
mkdir ./COCO17

wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lezhgY4M_Ag13w0dEzQ7x_zQ_w0ohjin' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lezhgY4M_Ag13w0dEzQ7x_zQ_w0ohjin" -O ./datasets/COCO17/annotations/instances_minitrain2017.json && rm -rf /tmp/cookies.txt

unzip train2017.zip -d ./COCO17
unzip val2017.zip -d ./COCO17
unzip test2017.zip -d ./COCO17
unzip annotations_trainval2017.zip -d ./COCO17

rm train2017.zip
rm val2017.zip
rm test2017.zip
rm annotations_trainval2017.zip

cd ../

