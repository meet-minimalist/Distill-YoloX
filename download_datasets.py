import os
import gdown

os.system("git clone https://meet-minimalist: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@github.com/meet-minimalist/Distill-YoloX.git")

ckpt_url = 'https://drive.google.com/drive/folders/1XceyVhvZY3HbRDB21qFsE10VS3NQUFJd?usp=sharing'
dataset_url = 'https://drive.google.com/drive/folders/1Expwv8Wa-RztONUzPXu9w2ByPjtcldno?usp=sharing'

gdown.download_folder(ckpt_url, output="./Distill-YoloX/", quiet=False, use_cookies=False)
gdown.download_folder(dataset_url, output="./Distill-YoloX/", quiet=False, use_cookies=False)

os.system("unzip ./Distill-YoloX/datasets/train2017.zip -d ./Distill-YoloX/datasets/")
os.system("unzip ./Distill-YoloX/datasets/val2017.zip -d ./Distill-YoloX/datasets/")
os.system("unzip ./Distill-YoloX/datasets/test2017.zip -d ./Distill-YoloX/datasets/")
os.system("unzip ./Distill-YoloX/datasets/annotations_trainval2017.zip -d ./Distill-YoloX/datasets/")

os.remove("./Distill-YoloX/datasets/train2017.zip")
os.remove("./Distill-YoloX/datasets/val2017.zip")
os.remove("./Distill-YoloX/datasets/test2017.zip")
os.remove("./Distill-YoloX/datasets/annotations_trainval2017.zip")
