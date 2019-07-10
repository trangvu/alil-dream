#!/bin/bash

#SBATCH --job-name=alner-policy
#SBATCH --account=da33
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50000
#SBATCH --gres=gpu:1
#SBATCH --partition=m3g
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

module load cuda/9.0
module load python/3.6.2
source $ROOT_DIR/../env/bin/activate
#module load tensorflow/1.12.0-python3.6-gcc5

export CUDA_VISIBLE_DEVICES=0
CACHE_PATH=/tmp/nv-$DATE
mkdir $CACHE_PATH
export CUDA_CACHE_PATH=$CACHE_PATH


DATASETS=( conll2003 multilingual multilingual multilingual)
EXPS=( en.bi en.es.de en.es.nl en.de.nl)
TRAINS=( en.train en.es.de.train en.es.nl.train en.de.nl.train )
TESTS=( en.testa en.es.de.testa en.es.nl.testa en.de.nl.testa )
DEVS=( en.testb en.es.de.testb en.es.nl.testb en.de.nl.testb )
VOCAB_SIZES=( 20000 35000 35000 35000 )
STRATEGIES=( Random Uncertainty Diversity )

index=$1
DATASET_NAME=${DATASETS[$index]}
EXP_NAME=${EXPS[$index]}
TRAIN_FILE=${TRAINS[$index]}
DEV_FILE=${DEVS[$index]}
TESTS_FILE=${TESTS[$index]}
EMBEDING_FILE=$DATA_DIR/"twelve.table4.multiCCA.window_5+iter_10+size_40+threads_16.normalized"
TEXT_DATA_DIR=$DATA_DIR'/'$DATASET_NAME
vocab=${VOCAB_SIZES[$index]}

OUTPUT=$OUT_DIR/${DATASET_NAME}_${EXP_NAME}_${DATE}
mkdir -p $OUTPUT
echo "TRAIN AL POLICY on dataset ${DATASET_NAME} experiment name ${EXP_NAME} "
cd $SRC_PATH && python AL-crf-simulation.py --root_dir $ROOT_DIR --dataset_name $DATASET_NAME  \
    --train_file $TEXT_DATA_DIR/$TRAIN_FILE --dev_file $TEXT_DATA_DIR/$DEV_FILE \
    --test_file $TEXT_DATA_DIR/$TESTS_FILE \
    --word_vec_file $EMBEDING_FILE --episodes 100 --timesteps 5 \
    --output $OUTPUT --label_data_size 100 --annotation_budget 200 \
    --initial_training_size 0 --vocab_size $vocab --ibo_scheme
rm -r -f $CACHE_PATH