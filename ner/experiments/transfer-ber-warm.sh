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
OUT_DIR=$ROOT_DIR/results/$DATE
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

POLICY_NAMES=( en.bi.policy  )
POLICY_PATHS=( conll2003_en.bi/conll2003_policy.ckpt )

EMBEDING_FILE=$DATA_DIR/"bio_nlp_vec/PubMed-shuffle-win-30.txt"

index=0
policy_idx=0
DATASET_NAME=${DATASETS[$index]}
EXP_NAME=${EXPS[$index]}
TRAIN_FILE=${TRAINS[$index]}
DEV_FILE=${DEVS[$index]}
TESTS_FILE=${TESTS[$index]}
POLICY_PATH=$ROOT_DIR/policy/${POLICY_PATHS[$policy_idx]}
POLICY_NAME=${POLICY_NAMES[$policy_idx]}
TEXT_DATA_DIR=$DATA_DIR'/'$DATASET_NAME
TAGGER_PATH=$ROOT_DIR/policy/conll2003_en_tagger.h5

OUTPUT=$OUT_DIR/transfer_warm_${DATASET_NAME}_${EXP_NAME}_${DATE}
mkdir -p $OUTPUT
echo "TRANSFER AL POLICY ${POLICY_NAME} with policy path ${POLICY_PATH} on dataset ${DATASET_NAME} experiment name ${EXP_NAME} "
cd $SRC_PATH && python AL-crf-transfer.py --root_dir $ROOT_DIR --dataset_name $DATASET_NAME  \
    --train_file $TEXT_DATA_DIR/$TRAIN_FILE --dev_file $TEXT_DATA_DIR/$DEV_FILE \
    --test_file $TEXT_DATA_DIR/$TESTS_FILE \
    --policy_path $POLICY_PATH\
    --word_vec_file $EMBEDING_FILE --episodes 1 --timesteps 20 \
    --output $OUTPUT --label_data_size 100 --annotation_budget 200 \
    --initial_training_size 0 --vocab_size 20000 \
    --ber_task --al_candidate_selection_mode random \
    --model_path $TAGGER_PATH
rm -r -f $CACHE_PATH