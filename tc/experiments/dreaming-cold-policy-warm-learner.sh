#!/bin/bash

#SBATCH --job-name=alil-dreaming
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

module load cuda/9.0
module load python/3.6.2
module load tensorflow/1.12.0-python3.6-gcc5

export CUDA_VISIBLE_DEVICES=0
CACHE_PATH=/tmp/nv-$DATE
mkdir $CACHE_PATH
export CUDA_CACHE_PATH=$CACHE_PATH


DATASETS=( elec_music book_movie ap17 ap17 )
EXPS=( target target es pt )
STRATEGIES=( Random Uncertainty Diversity )
DIMS=( 100 100 40 40 )
POLICY_PATHS=( elec_policy.h5 book_policy.h5 AP17en_policy.h5 AP17en_policy.h5 )
MODEL_PATHS=( elec_classifier.h5 book_classifier.h5 AP17en_classifier.h5 AP17en_classifier.h5 )

index=$1

POLICY_PATH=$ROOT_DIR'/policy/'${POLICY_PATHS[$index]}
MODEL_PATH=$ROOT_DIR'/policy/'${MODEL_PATHS[$index]}
ndream=5
dream_length=10
DATASET_NAME=${DATASETS[$index]}
EXP_NAME=${EXPS[$index]}
EMBEDING_FILE="${DATASET_NAME}_w2v.txt"
TEXT_DATA_DIR=$DATA_DIR'/'$DATASET_NAME'/'$EXP_NAME
TEST_SET=$DATA_DIR'/'$DATASET_NAME'/'$EXP_NAME'_test'
EMB_DIM=${DIMS[$index]}


echo "TRAIN AL POLICY on dataset ${DATASET_NAME} experiment name ${EXP_NAME} fold ${i}"
OUTPUT=$OUT_DIR/dreaming_cold_policy_warm_learner_${DATASET_NAME}_${EXP_NAME}_${DATE}/fold-${i}
mkdir -p $OUTPUT
cd $SRC_PATH && python ALIL-dreaming.py --root_dir $ROOT_DIR --dataset_name $DATASET_NAME --text_data_dir $TEXT_DATA_DIR \
    --word_vec_dir $DATA_DIR/word_vec/$EMBEDING_FILE --model_path $MODEL_PATH \
    --output $OUTPUT --embedding_dim $EMB_DIM --annotation_budget 100  \
    --test_set $TEST_SET --learning_phase_length 5 --initial_training_size 5 \
    --dreaming_budget $dream_length --ndream $ndream --timesteps 20 \
    --classifier_learning_rate 0.0001
rm -r -f $CACHE_PATH