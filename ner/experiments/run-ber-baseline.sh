#!/bin/bash

#SBATCH --job-name=alil-policy
#SBATCH --account=da33
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50000
#SBATCH --gres=gpu:1
#SBATCH --partition=m3h
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vuth0001@student.monash.edu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ROOT_DIR=`cd ..&&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR
DATA_DIR=$ROOT_DIR'/datadir'
OUT_DIR=$ROOT_DIR/results
mkdir -p $OUT_DIR

module load cuda/9.0
module load python/3.6.2
source $ROOT_DIR/../env/bin/activate

export CUDA_VISIBLE_DEVICES=0
CACHE_PATH=/tmp/nv-$DATE
mkdir $CACHE_PATH
export CUDA_CACHE_PATH=$CACHE_PATH


DATASETS=( genia4er )
EXPS=( ber )
TRAINS=( genia.train )
TESTS=( genia.test )
DEVS=( genia.dev )
STRATEGIES=( Random Uncertainty Diversity )

EMBEDING_FILE=$DATA_DIR/"bio_nlp_vec/PubMed-shuffle-win-30.txt"
index=0
QUERY_STRATEGY=$1
    DATASET_NAME=${DATASETS[$index]}
    EXP_NAME=${EXPS[$index]}
    TRAIN_FILE=${TRAINS[$index]}
    DEV_FILE=${DEVS[$index]}
    TESTS_FILE=${TESTS[$index]}
    TEXT_DATA_DIR=$DATA_DIR'/'$DATASET_NAME
set -x
TAGGER_PATH=$ROOT_DIR/policy/conll2003_en_tagger.h5
#    for QUERY_STRATEGY in "${STRATEGIES[@]}"; do
        OUTPUT=$OUT_DIR/${DATASET_NAME}_${EXP_NAME}_${STRATEGIES[$QUERY_STRATEGY]}_${DATE}
        mkdir -p $OUTPUT
        echo "RUN BASELINE ON dataset ${DATASET_NAME} experiment name ${EXP_NAME} query strategy ${STRATEGIES[$QUERY_STRATEGY]}"
        cd $SRC_PATH && python AL-crf-baselines.py --root_dir $ROOT_DIR --dataset_name $DATASET_NAME \
            --train_file $TEXT_DATA_DIR/$TRAIN_FILE --dev_file $TEXT_DATA_DIR/$DEV_FILE \
            --test_file $TEXT_DATA_DIR/$TESTS_FILE \
            --word_vec_file $EMBEDING_FILE --timesteps 20 --ber_task \
            --output $OUTPUT --annotation_budget 200 \
            --initial_training_size 0 --vocab_size 20000 --query_strategy ${STRATEGIES[$QUERY_STRATEGY]} \
            --model_path $TAGGER_PATH
#    done

rm -r -f $CACHE_PATH