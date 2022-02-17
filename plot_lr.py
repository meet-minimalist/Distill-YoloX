from yolox.utils import LRScheduler
from tqdm import tqdm

import matplotlib.pyplot as plt

max_epoch = 60
basic_lr_per_img = 0.01 / 64.0
scheduler_name = "yoloxwarmcos"
scheduler_name = "warmcos"
semi_epoch = 30
warmup_epochs = 5
warmup_lr = 0
no_aug_epochs = 15
min_lr_ratio = 0.05
batch_size = 16
max_iter = 1563

scheduler = LRScheduler(
    scheduler_name,
    basic_lr_per_img * batch_size, 
    max_iter,
    max_epoch,
    semi_epoch=semi_epoch,
    warmup_epochs=warmup_epochs,
    warmup_lr_start=warmup_lr,
    no_aug_epochs=no_aug_epochs,
    min_lr_ratio=min_lr_ratio,
)

lr_values = []
step_values = []
for e in tqdm(range(max_epoch)):
    for c in range(max_iter):
        step = e * max_iter + c + 1
        lr = scheduler.update_lr(step)

        lr_values.append(lr)
        step_values.append(step)

 
 
# Plotting the Graph
plt.plot(step_values, lr_values)
plt.title("Curve plotted using the given points")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()