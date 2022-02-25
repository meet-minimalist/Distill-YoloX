

mkdir ./pretrained_ckpt
mkdir ./datasets
mkdir ./datasets/VOC0712

wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -O ./pretrained_ckpt/yolox_s.pth
wget -c https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth -O ./pretrained_ckpt/yolox_l.pth


wget -c https://pjreddie.com/media/files/VOC2012test.tar
wget -c https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget -c https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget -c http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

tar -xvf VOC2012test.tar -C ./datasets/VOC0712
tar -xvf VOCtrainval_11-May-2012.tar -C ./datasets/VOC0712
tar -xvf VOCtrainval_06-Nov-2007.tar -C ./datasets/VOC0712
tar -xvf VOCtest_06-Nov-2007.tar -C ./datasets/VOC0712

mv ./datasets/VOC0712/VOC2012test/VOCdevkit/VOC2012 ./datasets/VOC0712/VOC2012/
mv ./datasets/VOC0712/VOCtrainval_11-May-2012/VOCdevkit/VOC2012 ./datasets/VOC0712/VOC2012/
mv ./datasets/VOC0712/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2012 ./datasets/VOC0712/VOC2012/
mv ./datasets/VOC0712/VOCtest_06-Nov-2007/VOCdevkit/VOC2012 ./datasets/VOC0712/VOC2012/

rm VOC2012test.tar
rm VOCtrainval_11-May-2012.tar
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar

