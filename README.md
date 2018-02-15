# HappyDogYear

使用`Inception V3`进行狗狗图片分类，祝大家狗年大吉！

## 数据集

数据集来自`Stanford Dog Dataset`

[http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar)

共包括120种狗狗的20580张图片，下载后解压得到`Images`文件夹，其中包括120个子文件夹，每个子文件夹对应一种狗狗

## 模型训练

使用`Inception V3`和狗狗数据进行迁移训练，得到能够识别狗狗类别的图片分类模型

迁移训练的代码来自`TensorFlow`官方Github

[https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py)

使用以下命令即可进行训练，当然也可以直接使用我训练好的模型

```
python retrain.py --image_dir Images/ --output_graph dog_inception_v3_graph.pb --output_labels dog_inception_v3_labels.txt --summaries_dir summaries_dir --model_dir . --bottleneck_dir bottleneck_dir
```

## 模型使用

迁移训练得到两个文件

- `dog_inception_v3_graph.pb`：狗狗图片分类模型
- `dog_inception_v3_labels.txt`：全部的狗狗类别

使用`classify_single_dog.py`即可进行单张图片的分类，其中最后一行的`test.jpg`即为待分类图片的路径，这里提供了一张京巴犬的图片

## 其他文件

- `classify_all_dogs.py`：将`Images`文件夹中的20580张图片全部分类，并将正确标签以及概率前五的分类结果和对应概率写入`results.txt`，便于后续统计
- `results.txt`：即上面提到的分类结果文件
- `stats.py`：根据`results.txt`进行一些简单的统计，包括每种狗狗的分类正确率、不同狗狗之间的关联和对比等
- `sample.py`：从每类狗狗中随机抽取一张，共120张图片并生成一张拼图