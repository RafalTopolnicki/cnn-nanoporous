#!/bin/bash

# train model without descriptors
python train.py --outputdir results/ --database database_withfeatures.csv \
        --lr 0.001 --optimizer Adam --batch_size 40  --model_name densenet-201 \
        --p_flip 0.3 --aug_roll_ratio 0.3 --cv_folds 8 --folds 0 1 2 3 4 5 6 7 \
        --suffix 5k_flip0.3roll0.3_strat_nodesc --use_stratification --size 80

# train model that uses the descriptors
python train.py --outputdir results/ --database database.csv \
         --lr 0.001 --optimizer Adam --batch_size 40  --model_name densenet-201 \
         --p_flip 0.3 --aug_roll_ratio 0.3 --cv_folds 8 --folds 0 1 2 3 4 5 6 7 \
         --suffix 5k_flip0.3roll0.3_strat_withdesc --use_stratification  --size 80 \
         --use_descriptors --fc_first_size 33 33 21 --fc_second_size 32 32 32 --p_dropout 0.0
