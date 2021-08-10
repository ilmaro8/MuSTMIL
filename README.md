# MuSTMIL
Implementation of MuSTMIL (Multi-Scale Task Multiple Instance Learning).

## Reference
If you find this repository useful in your research, please cite:

[1] 

Paper link: 

## Requirements
Python==3.6.9, albumentations==0.1.8, numpy==1.17.3, opencv==4.2.0, pandas==0.25.2, pillow==6.1.0, torchvision==0.8.1, pytorch==1.7.0

## Best models
- The best model for the MuSTMIL CNN is available [here](...).

## Pre-Processing
The WSIs are split in 224x224 pixels patches, from magnification 5/10x. 
The methods used to extract the patches come from [Multi_Scale_Tools library](https://github.com/sara-nl/multi-scale-tools)

The method is in the /preprocessing folder of the Multi_Scale_Tools library: 
- python Patch_Extractor_Dense_Grid.py -m 10 -w 1.25 -p 10 -r True -s 224 -x 0.7 -y 0 -i /PATH/CSV/IMAGES/TO/EXTRACT.csv -t /PATH/TISSUE/MASKS/TO/USE/ -o /FOLDER/WHERE/TO/STORE/THE/PATCHES/

The method to link patches from different magnification levels is in: /classification/Generate_csv_upper_region.py

More info: https://www.frontiersin.org/articles/10.3389/fcomp.2021.684521/full

## CSV Input Files:
CSV files are used as input for the scripts. The csvs have the following structures
- For each partition (train, validation, test), the csv file has id_img, cancer, high-grade dysplasia, low-grade dysplasia, hyperplastic polyp, normal glands as column.

## Training
Script to train the CNN at WSI-level, using an embedding-based MIL CNN:
### Single Scale
- python training_single_scale.py -c resnet34 -b 512 -p att -e 10 -t multilabel -f True -m 10 -i /PATH/WHERE/TO/FIND/THE/CSVS/INCLUDING/THE/PARTITIONS -o /PATH/WHERE/TO/SAVE/THE/MODEL/WEIGHTS -w /PATH/WHERE/TO/FIND/THE/PATCHES
### Multi Scale
- python training_multi_scale.py -c resnet34 -b 512 -p att -e 10 -t multilabel -f True -m 10_5 -i /PATH/WHERE/TO/FIND/THE/CSVS/INCLUDING/THE/PARTITIONS -o /PATH/WHERE/TO/SAVE/THE/MODEL/WEIGHTS -w /PATH/WHERE/TO/FIND/THE/PATCHES

## Testing
### Single Scale
- python testing_single_scale_WSI.py -c resnet34 -b 512 -p att -t multilabel -f True -w /PATH/TO/MODEL/WEIGHTS.pt -i /PATH/TO/INPUT/CSV.csv -w /PATH/WHERE/TO/FIND/THE/PATCHES
### Multi Scale
Multi-scale output of the MuSTMIL CNN
- python testing_multi_scale_WSI.py -c resnet34 -b 512 -p att -t multilabel -f True -w /PATH/TO/MODEL/WEIGHTS.pt -i /PATH/TO/INPUT/CSV.csv -w /PATH/WHERE/TO/FIND/THE/PATCHES -m 10_5
### Single Scale on Multi-scale branches
Single-scale output of the MuSTMIL CNN
- python testing_multi_scale_on_single_scale_WSI.py -c resnet34 -b 512 -p att -t multilabel -f True -m /PATH/TO/MODEL/WEIGHTS.pt -i /PATH/TO/INPUT/CSV.csv -w /PATH/WHERE/TO/FIND/THE/PATCHES -m 10

## Acknoledgements
This project has received funding from the EuropeanUnion’s Horizon 2020 research and innovation programme under grant agree-ment No. 825292 [ExaMode](http://www.examode.eu). Infrastructure fromthe SURFsara HPC center was used to train the CNN models in parallel. 
