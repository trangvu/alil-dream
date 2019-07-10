# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 11:05:31 2017

@author: lming
"""
import os
from shutil import copyfile

from query_strategy import randomSample, uncertaintySample, diversitySample, diversityallSample
from tagger import CRFTagger
import utilities
import tensorflow as tf
import numpy as np

args = utilities.get_args()
logger = utilities.init_logger()

max_len = args.max_seq_length
VOCABULARY= args.vocab_size
EPISODES=args.timesteps
BUDGET=args.annotation_budget
k=args.k
rootdir=args.root_dir
train_file=args.train_file
dev_file=args.dev_file
test_file=args.test_file
emb_file= args.word_vec_file
QUERY=QUERY=args.query_strategy
DATASET_NAME=args.dataset_name

tagger_output = "{}/{}_tagger.h5".format(args.output, DATASET_NAME)
tagger_temp = "{}/{}_tagger_temp.h5".format(args.output, DATASET_NAME)
resultname = "{}/{}_accuracy.txt".format(args.output, DATASET_NAME)

logger.info("Baseline AL with budget {}".format(DATASET_NAME, BUDGET))
logger.info(" * Query strategy: {}".format(QUERY))
logger.info(" * Vocabulary size: {}".format(VOCABULARY))
logger.info(" * INPUT train file: {}".format(train_file))
logger.info(" * INPUT dev file: {}".format(dev_file))
logger.info(" * INPUT test file: {}".format(test_file))
logger.info(" * TAGGER OUTPUT file: {}".format(tagger_output))
logger.info(" * ACC OUTPUT file: {}".format(resultname))

if args.ber_task:
    logger.info("Processing data for BER task")
    train_x, train_y, train_lens = utilities.load_data2labels_BER(train_file)
    test_x, test_y, test_lens = utilities.load_data2labels_BER(test_file)
    dev_x, dev_y, dev_lens = utilities.load_data2labels_BER(dev_file)
    label2str = utilities.BER_label2str
    num_classes = 6
elif args.ibo_scheme:
    logger.info("Processing data - IBO scheme")
    train_x, train_y, train_lens = utilities.load_data2labels_IBO(train_file)
    test_x, test_y, test_lens = utilities.load_data2labels_IBO(test_file)
    dev_x, dev_y, dev_lens = utilities.load_data2labels_IBO(dev_file)
    label2str = utilities.IBO_label2str
    num_classes = 9
else:
    logger.info("Processing data - IO scheme")
    train_x, train_y, train_lens = utilities.load_data2labels_IO(train_file)
    test_x, test_y, test_lens = utilities.load_data2labels_IO(test_file)
    dev_x, dev_y, dev_lens = utilities.load_data2labels_IO(dev_file)
    label2str = utilities.IO_label2str
    num_classes = 5

train_sents = utilities.data2sents(train_x, train_y)
test_sents = utilities.data2sents(test_x, test_y)
dev_sents = utilities.data2sents(dev_x, dev_y)

# build vocabulary
logger.info("Max document length: {}".format(str(max_len)))
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_document_length=max_len, min_frequency=1)
# vocab = vocab_processor.vocabulary_ # start from {"<UNK>":0}
train_idx = list(vocab_processor.fit_transform(train_x))
dev_idx = list(vocab_processor.fit_transform(dev_x))
vocab = vocab_processor.vocabulary_
vocab.freeze()
test_idx = list(vocab_processor.fit_transform(test_x))

# build embeddings
vocab = vocab_processor.vocabulary_
vocab_size = VOCABULARY
w2v = utilities.load_crosslingual_embeddings(emb_file, vocab, vocab_size)

allf1list=[]  
#Training the policy
for tau in range(0, 20):
    f1list=[]
    # Shuffle train_sents, split into train_la and train_pool
    indices = np.arange(len(train_sents))
    np.random.shuffle(indices)
    train_la = []
    train_pool = []
    train_la_idx = []
    train_pool_idx = []
    for i in range(0, len(train_sents)):
        if (i < args.initial_training_size):
            train_la.append(train_sents[indices[i]])
            train_la_idx.append(train_idx[indices[i]])
        else:
            train_pool.append(train_sents[indices[i]])
            train_pool_idx.append(train_idx[indices[i]])
    # Initialise the model
    tagger_output = "{}/{}_tagger.h5".format(args.output, DATASET_NAME)
    if os.path.exists(tagger_output):
        os.remove(tagger_output)
    if args.model_path is not None:
        copyfile(args.model_path, tagger_output)
    model = CRFTagger(tagger_output, num_classes=num_classes)
    if args.initial_training_size > 0:
        model.train(train_la)

    for t in range(0, BUDGET):
        logger.info('Episode: '+str(tau+1)+' Budget so far: '+str(t+1))
        # random_pool, random_pool_idx, queryindices = utilities.randomKSamples(train_pool, train_pool_idx, args.k)
        if (QUERY == 'Random'):
            logger.info("Random sampling")
            sampledata, samplelabels, train_pool, train_pool_idx = randomSample(train_pool, train_pool_idx, 1)
        elif (QUERY == 'Uncertainty'):
            logger.info("Uncertainty sampling")
            sampledata, samplelabels, train_pool, train_pool_idx = uncertaintySample(train_pool, train_pool_idx, 1, model)
        elif (QUERY == 'Diversity'):
            logger.info("Diversity sampling")
            sampledata, samplelabels, train_pool, train_pool_idx = diversityallSample(train_pool, train_pool_idx, 1, train_la)

        train_la = train_la + sampledata
        train_la_idx = train_la_idx + samplelabels
        model.train(train_la)
        
        if((t+1) % 5 == 0):
            f1score=model.test(test_sents, label2str)
            f1list.append(f1score)
            logger.info(f1score)
        allf1list.append(f1list)

    f1array=np.array(allf1list)
    averageacc=list(np.mean(f1array, axis=0))
    logger.info('F1 list: ')
    logger.info(averageacc)
    ww=open(resultname,'w')
    ww.writelines(str(line)+ "\n" for line in averageacc)
    ww.close()
logger.info(resultname)