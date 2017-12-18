# sonnet_xception

## getting start
### install
for CPU environments,
```console
pip install -r requirements_cpu.txt
```

for GPU environments,
```console
pip install -r requirements_gpu.txt
```

### get data for training
You can get 17 Flower Dataset from [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/).

### execute training
Set image data directory as name 'jpg' and then,
```console
python train_xception.py
```
trained model and some data for TensorBoard are saved to `summary/YYYY_mm_dd_HH_MM_SS/`.

### TensorBoard
```console
tensorboard --logdir summary/YYYY_mm_dd_HH_MM_SS
```

