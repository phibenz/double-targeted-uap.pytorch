#### Train resnet20 on CIFAR10 160 epochs#####
python3 train_model.py \
  --pretrained_dataset cifar10 --pretrained_arch resnet20 \
  --epochs 160 --batch_size 128 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 4 --ngpu 1

#### Train vgg16 on CIFAR10 160 epochs#####
python3 train_model.py \
  --pretrained_dataset cifar10 --pretrained_arch vgg16_cifar \
  --epochs 160 --batch_size 128 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 10 --ngpu 1

#### Train ResNet56 on CIFAR100 160 epochs#####
python3 train_model.py \
  --pretrained_dataset cifar100 --pretrained_arch resnet56 \
  --epochs 160 --batch_size 128 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 10 --ngpu 1

#### Train vgg19 on CIFAR100 160 epochs#####
python3 train_model.py \
  --pretrained_dataset cifar100 --pretrained_arch vgg19_cifar \
  --epochs 160 --batch_size 128 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 10 --ngpu 1

#### Train VGG16 on GTSRB 160 epochs #####
python3 train_model.py \
  --pretrained_dataset gtsrb --pretrained_arch vgg16_cifar \
  --epochs 160 --batch_size 128 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 4 --ngpu 1

#### Train resnet20 on GTSRB 160 epochs #####
python3 train_model.py \
  --pretrained_dataset gtsrb --pretrained_arch resnet20 \
  --epochs 160 --batch_size 128 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 4 --ngpu 1

#### Train resnet50 on EUROSAT 160 epochs #####
python3 train_model.py \
  --pretrained_dataset eurosat --pretrained_arch resnet50 \
  --epochs 160 --batch_size 32 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 4 --ngpu 1

#### Train Inception V3 on EUROSAT 160 epochs #####
python3 train_model.py \
  --pretrained_dataset eurosat --pretrained_arch inception_v3 \
  --epochs 160 --batch_size 32 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 4 --ngpu 1

### Train resnet50 V3 on YCB 160 epochs #####
python3 train_model.py \
  --pretrained_dataset ycb --pretrained_arch resnet50 \
  --epochs 60 --batch_size 32 --learning_rate 0.1 --decay 1e-4 \
  --schedule 20 40 --gammas 0.1 0.1  --workers 6 --ngpu 1

### Train Inception V3 on YCB 160 epochs #####
python3 train_model.py \
  --pretrained_dataset ycb --pretrained_arch inception_v3 \
  --epochs 60 --batch_size 32 --learning_rate 0.1 --decay 1e-4 \
  --schedule 20 40 --gammas 0.1 0.1  --workers 6 --ngpu 1

### Train vgg16 V3 on YCB 160 epochs #####
python3 train_model.py \
  --pretrained_dataset ycb --pretrained_arch vgg16 \
  --epochs 30 --batch_size 32 --learning_rate 0.001 --decay 1e-4 \
  --workers 6 --ngpu 1

### Train MobileNet V2 on YCB 160 epochs #####
python3 train_model.py \
  --pretrained_dataset ycb --pretrained_arch mobilenet_v2 \
  --epochs 60 --batch_size 32 --learning_rate 0.1 --decay 1e-4 \
  --schedule 20 40 --gammas 0.1 0.1  --workers 6 --ngpu 1
