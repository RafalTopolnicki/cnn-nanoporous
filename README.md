# cnn-nanoporous
## Transferable 3D Convolutional Neural Networks for Elastic Constants Prediction in Nanoporous  Metals

This repository contain codebase for the paper 
*Transferable 3D Convolutional Neural Networks for Elastic Constants Prediction in Nanoporous Metals*

### Nanoporous structures generation 
Three-dimensional periodic bicontinuous nanoporous structures were generated based on the method proposed by [Soyarslan et al.](https://doi.org/10.1016/j.actamat.2018.01.005), which involves superposition of standing sinusoidal waves with fixed wavelengths and varying phase. 

Three datasets generated in this study are provided in the `Datasets` folder:

- **Dataset 1**: Contains 5000 nanoporous gold structures with a target solid volume fraction of 0.25.
- **Dataset 2**: Contains 1000 NPG structures with a target solid volume fraction of 0.35.
- **Dataset 3**: Contains 422 nanoporous silver structures with MD-computed elastic constants. It is divided into two parts: structures 1–250 and 251–422 have target solid volume fractions of 0.25 and 0.35, respectively.

Each structure in the datasets is labeled with Molecular Dynamics (MD) computed elastic constants in three primcipal directions, and a set of computed morphological and topological descriptors. All structures are accompanied by 30 phase values, required for identical regeneration. After the atomistic structure is constructed in `.lmp` format, it is converted into a binary 3D array with the specified grid resolution.

The generator works with the following command
```
python grid_generator.py --dataset Datasets/Dataset1_gold_0.25.xlsx --grid_resolution 80 --outputdir Datasets/results --keep_lmp_files 

```

### Model training
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


### Credits
Some CNN implementations used in this repo modifies the code from [xmuyzz/3D-CNN-PyTorch](https://github.com/xmuyzz/3D-CNN-PyTorch) repository. 
 
