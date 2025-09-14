# cnn-nanoporous
## Transferable 3D Convolutional Neural Networks for Elastic Constants Prediction in Nanoporous  Metals

This repository contain codebase for the paper 
*Transferable 3D Convolutional Neural Networks for Elastic Constants Prediction in Nanoporous Metals*

### Setup
To train the model one need to provide the:
   1. dataset that include all the structures, each kept in separate `.npy` file (binary file format used by NumPy to efficiently store arrays).
   2. a `.csv` file that for each structure provide a mapping between path to its file (column name `npy_path`), the value of elastic constant stored in column (`cii`) and optionally list of additional descriptors
```
npy_path	cii	Genus	Solid Volume Fraction	Number of Nodes	Mean Nodes Diameter	Nodes Diameter Standard Deviation	Mean Ligaments Diameter	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/0_rot-x.npy	0.763035	6	0.265	28	75.809	13.066	60.165	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/1_rot-x.npy	1.505392	5	0.275	30	77.56	14.807	55.442	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/2_rot-x.npy	1.511555	4	0.267	30	73.027	14.271	64.65	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/3_rot-x.npy	0.341111	5	0.233	19	71.369	15.981	67.228	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/4_rot-x.npy	0.358744	1	0.243	19	77.502	12.793	63.799	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/5_rot-x.npy	2.476169	9	0.28	31	77.736	11.347	68.959	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/6_rot-x.npy	2.092207	7	0.258	32	67.656	16.139	59.48	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/7_rot-x.npy	0.413300	3	0.24	20	72.718	14.825	68.353	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/8_rot-x.npy	0.076880	2	0.222	21	68.661	14.375	62.859	….
```
Then all it takes is to run the following command (see `run_training.sh`), providing that directory `results` exists
```
python train.py --outputdir results/ --database database_withfeatures.csv \
        --lr 0.001 --optimizer Adam --batch_size 40  --model_name densenet-201 \
        --p_flip 0.3 --aug_roll_ratio 0.3 --cv_folds 8 --folds 0 1 2 3 4 5 6 7 \
        --suffix 5k_flip0.3roll0.3_strat_nodesc --use_stratification --size 80
```
Use argument `--model_name` to specify CNN architecture to use. Options are
```
    efficientnet-b0
    efficientnet-b4
    densenet-121
    densenet-169
    densenet-201
    densenet-264
    resnet-18
    resnet-50
    resnet-101
    cnn
    mobilenet
```

## Running Inference
The champion DenseNet-201 model is available in the models/ directory. 
There are 40 exemplary voxelized structures provided in the `exemplary_structures/` directory.
You can run inference directly using the provided model **without any training** (GPU is not required).
To predict the Elastic Modulus for these 40 structures, run:
```
python inference.py --outputfile inference_results.csv --model_path models/model_DenseNet201.pth  --txt_path inference_filelist.txt
```
The predictions will be saved to the `inference_results.csv` file, which will look like this:
```
path,prediction
exemplary_structures/155_rot-z.npy,0.92881143
exemplary_structures/1689_rot-z.npy,0.35452503
exemplary_structures/2599_rot-z.npy,0.701846
exemplary_structures/2193_rot-z.npy,1.0405514
exemplary_structures/3678_rot-y.npy,0.9879305
exemplary_structures/3217_rot-y.npy,0.4050305
[....]
```
The expected predicted Elastic Modulus values, along with the exact values obtained using MD, are available in the `expected_inference_results.csv` file.
The R2 coefficient for these 40 exemplary structures is **0.9488**.

### Credits
Some CNN implementations used in this repo modifies the code from [xmuyzz/3D-CNN-PyTorch](https://github.com/xmuyzz/3D-CNN-PyTorch) repository. 
 
