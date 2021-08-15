## 简介
本项目是在[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)基础上添加PGD,Free,FGSM,三种对抗扰动方案
，只需要clone本项目
## 环境
python 3.7  
pytorch 1.6  
tqdm  
sklearn  
tensorboardX 

## 实验方法
### 训练
baseline:
python run.py --model TextCNN --mode train 

PGD:
python run.py --model TextCNN --mode train --attack pgd --epsilon 0.8 --alpha 0.2 --attack_iter 5

Free:
python run.py --model TextCNN --mode train --attack free --epsilon 0.8 --attack_iter 5

FGSM:
python run.py --model TextCNN --mode train --epsilon 0.01

  
### 测试  
另外本人已经将model上传至git,可以直接测试加以验证  
python run.py --model TextCNN --mode test  
python run.py --model TextCNN --mode test --attack pgd  
python run.py --model TextCNN --mode test --attack free  
python run.py --model TextCNN --mode test --attack fgsm     


