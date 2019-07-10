#!/bin/bash

#SBATCH --job-name=alner-transfer
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


DATASETS=( conll2003 conll2002 conll2002 conll2002)
EXPS=( en.bi es de nl)
TRAINS=( en.train es.train de.train nl.train )
TESTS=( en.testa es.testa de.testa nl.testa )
DEVS=( en.testb es.testb de.testb nl.testb )
VOCAB_SIZES=( 20000 20000 20000 20000 )

index=$1
DATASET_NAME=${DATASETS[$index]}
EXP_NAME=${EXPS[$index]}
TRAIN_FILE=${TRAINS[$index]}
DEV_FILE=${DEVS[$index]}
TESTS_FILE=${TESTS[$index]}
EMBEDING_FILE=$DATA_DIR/"twelve.table4.multiCCA.window_5+iter_10+size_40+threads_16.normalized"
TEXT_DATA_DIR=$DATA_DIR'/'$DATASET_NAME

#init_size=$1
OUTPUT=$OUT_DIR/dreaming_${DATASET_NAME}_${EXP_NAME}_${DATE}
mkdir -p $OUTPUT
echo "DREAM TRANSFER AL POLICY ${POLICY_NAME} with policy path ${POLICY_PATH} on dataset ${DATASET_NAME} experiment name ${EXP_NAME} "
cd $SRC_PATH && python AL-tagger-qual.py --root_dir $ROOT_DIR --dataset_name $DATASET_NAME  \
    --train_file $TEXT_DATA_DIR/$TRAIN_FILE --dev_file $TEXT_DATA_DIR/$DEV_FILE \
    --test_file $TEXT_DATA_DIR/$TESTS_FILE \
    --word_vec_file $EMBEDING_FILE --episodes 1 --timesteps 20 \
    --output $OUTPUT --label_data_size 100 --annotation_budget 200 \
    --initial_training_size 0 --learning_phase_length 5 --vocab_size 20000 \
    --ndream 5 --dreaming_budget 10 --initial_training_size 500
rm -r -f $CACHE_PATH