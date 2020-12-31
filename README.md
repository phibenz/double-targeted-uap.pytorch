# Double Targeted Universal Adversarial Perturbations
This is the repository for our ACCV 2020 paper titled [Double Targeted Universal Adversarial Perturbations](https://arxiv.org/pdf/2010.03288.pdf)

## Abstract
Despite their impressive performance, deep neural networks (DNNs) are widely known to be vulnerable to adversarial attacks, which makes it challenging for them to be deployed in security-sensitive applications, such as autonomous driving. Image-dependent perturbations can fool a network for one specific image, while universal adversarial perturbations are capable of fooling a network for samples from all classes without selection. We introduce a double targeted universal adversarial perturbations (DT-UAPs) to bridge the gap between the instancediscriminative image-dependent perturbations and the generic universal perturbations. This universal perturbation attacks one targeted source class to sink class, while having a limited adversarial effect on other nontargeted source classes, for avoiding raising suspicions. Targeting the source and sink class simultaneously, we term it double targeted attack (DTA). This provides an attacker with the freedom to perform precise attacks on a DNN model while raising little suspicion. We show the effectiveness of the proposed DTA algorithm on a wide range of datasets and also demonstrate its potential as a physical attack.

## Setup 
We performed our experiments with `PyTorch v.0.4.1`

### Config
Copy `config/sample_config.py ` to `config/config.py`. Edit the paths in `config/config.py` according to your system. 

### Datasets 
#### ImageNet
 1. Follow the common setup to make ImageNet compatible with pytorch as described in [here](https://github.com/pytorch/examples/tree/master/imagenet)
 2. Set the path to the pytorch ImageNet dataset folder in the config file
#### GTSRB
 1. Download `GTSRB-Training_fixed.zip, GTSRB_Final_Test_GT.zip, GTSRB_Final_Test_Images.zip` from [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).
 2. Extract the files into the following Folder structure:
```
GTSRB
    ∟- Training
    ∟- Final_Test
    ∟- GT-final_test.csv
```
 3. Set the path to the GTSRB folder in the config file
 4. Run `python3 ./dataset_utils/gtsrb_preparation.py`. This should generate a "Testing" folder in your GTSRB folder. The dataset is now ready to be used.

#### EuroSAT
 1. Dowload the EuroSat data from [here](madm.dfki.de/files/sentinel/EuroSAT.zip)
 2. Extract the zip file `unzip EuroSAT.zip`
 3. Set the path to the EuroSAT folder in the config file

#### YCB
1. Download the YCB dataset via `python3 dataset_utils/ycb_downloader.py`
2. Make a correction in the dataset with `mv ./ycb/home/bcalli/025_mug ./ycb/`.

#### Mobilenet V2
1. The Mobilenet V2 checkpoint was downloaded from [here](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth) and is placed in `./models/mobilenet_v2/mobilenet_v2-b0353104.pth`.
2. Modify the path to the checkpont accordingly. 

## Experiments
The code can be directly tested with `bash ./experiments/one2one_cifar10`.
To train additional models, have a look into the bash script `./experiments/train_model.sh`

## Citation
```
@inproceedings{benz2020double,
  title={Double targeted universal adversarial perturbations},
  author={Benz, Philipp and Zhang, Chaoning and Imtiaz, Tooba and Kweon, In So},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  year={2020}
}
```