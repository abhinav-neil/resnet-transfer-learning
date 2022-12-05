# Transfer Learning for Image Classification

## 1. Introduction
We perform transfer learning by fine-tuning a Resnet-18 model trained on the [ImageNet1k](https://huggingface.co/datasets/imagenet-1k) dataset, to classify images on the CIFAR-100 dataset. 

The ResNet is a neural network for image classification as described in the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). The ResNet-18 model is a 18-layer ResNet model pretrained on the ImageNet-1k dataset. The model is trained on 1000 classes of images and has an input image size of (3 x 224 x 224).

The [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60000 32x32 colour images in 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. 

## 2. Methodology

### 2.1 Comparison of pretrained models
We compare the following pretrained models (pretrained on ImageNet-1k) for their inference speed, top-1 accuracy and # of parameters:

- [VGG11](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html)

- [VGG11_BatchNorm](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg11_bn.html)
- [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- [ResNet34](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html)
- [DenseNet121](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html)
- [MobileNetV3_Small](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html)] 

(Code & results in [this notebook](./code/model_comparison.ipynb))

### 2.2 Fine-tuning the Resnet-18 model
We download the pretrained Resnet18 model from [PyTorch Hub](https://pytorch.org/hub/pytorch_vision_resnet/). We then fine-tune the model on the CIFAR-100 dataset by freezing all the layers except the last fully connected layer. We train the model for 10 epochs and use a learning rate of 0.001 and batch size of 64. We use the Adam optimizer for training. We use the cross-entropy loss function. We save the model with the best validation accuracy. We then evaluate the model on the test set.

```python
#train
code_dir=./
epochs=10
python $code_dir/train.py --epochs $epochs
# evaluate
python $code_dir/train.py --evaluate --checkpoint_name <checkpoint_name>
```

Test accuracy (top-1): 58.8%

### 2.3 Using augmentations
We use augmentations to improve robustness and generalization. We use the following augmentations: HorizontalFlip, GaussianBlur, RandomGrayScale and train the model for 10 epochs for each augmentation. We then evaluate the model on the test set. 

```python
augmentation=HorizontalFlip # HorizontalFlip, GaussianBlur, RandomGrayScale, None
# train
python $code_dir/train.py --epochs $epochs --augmentation $augmentation
# evaluate
python $code_dir/train.py --evaluate --checkpoint_name <checkpoint_name>
```
The best test accuracy (top-1) is 59.12% for the HorizontalFlip augmentation.

### 2.4 Robustness
We test the model for robustness by adding Gaussian noise to the test set. We add noise to the test set with a mean 0 and standard deviation of 0.01. We then evaluate the model on the noisy test set. 

```python
python $code_dir/train.py --evaluate --checkpoint_name <checkpoint_name> --test_noise
```

The test accuracy (top-1) is 25.4%, indicating that the model is not robust to Gaussian noise.