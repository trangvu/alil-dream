# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 11:05:31 2017

@author: lming
"""
from shutil import copyfile

from alagent import ALAgent
from tagger import CRFTagger
import utilities
import tensorflow as tf
import numpy as np
import time

args = utilities.get_args()
logger = utilities.init_logger()

max_len = args.max_seq_length
VOCABULARY= args.vocab_size
EPISODES=args.episodes
BUDGET=args.annotation_budget
k=args.k

rootdir=args.root_dir
train_file=args.train_file
dev_file=args.dev_file
test_file=args.test_file
emb_file= args.word_vec_file
DATASET_NAME=args.dataset_name
policy_path=args.policy_path
policy_output="{}/{}_policy.h5".format(args.output, DATASET_NAME)
tagger_output="{}/{}_tagger.h5".format(args.output, DATASET_NAME)
tagger_temp = "{}/{}_tagger_temp.h5".format(args.output, DATASET_NAME)
resultname= "{}/{}_accuracy.txt".format(args.output, DATASET_NAME)

logger.info("Transfer AL policy on dataset {} in {} episode with budget {}".format(DATASET_NAME, EPISODES, BUDGET))
logger.info(" * Vocabulary size: {}".format(VOCABULARY))
logger.info(" * INPUT policy: {}".format(policy_path))
logger.info(" * INPUT train file: {}".format(train_file))
logger.info(" * INPUT dev file: {}".format(dev_file))
logger.info(" * INPUT test file: {}".format(test_file))
logger.info(" * POLICY OUTPUT path: {}".format(policy_output))
logger.info(" * TAGGER OUTPUT file: {}".format(tagger_output))
logger.info(" * ACC OUTPUT file: {}".format(resultname))


if args.ber_task:
    logger.info("Processing data for BER task")
    train_x, train_y, train_lens = utilities.load_data2labels_BER(train_file)
    test_x, test_y, test_lens = utilities.load_data2labels_BER(test_file)
    dev_x, dev_y, dev_lens = utilities.load_data2labels_BER(dev_file)
    label2str = utilities.BER_label2str
    selected_modules = ["policy_net", "marginal_prob_cnn", "entropy_cnn", "entropy_embedding"]
    num_classes = 6
elif args.ibo_scheme:
    logger.info("Processing data - IBO scheme")
    train_x, train_y, train_lens = utilities.load_data2labels_IBO(train_file)
    test_x, test_y, test_lens = utilities.load_data2labels_IBO(test_file)
    dev_x, dev_y, dev_lens = utilities.load_data2labels_IBO(dev_file)
    label2str = utilities.IBO_label2str
    selected_modules = ["sentence_cnn","marginal_prob_cnn","labeled_pool","policy_net", "entropy_cnn", "entropy_embedding"]
    num_classes = 9
else:
    logger.info("Processing data - IO scheme")
    train_x, train_y, train_lens = utilities.load_data2labels_IO(train_file)
    test_x, test_y, test_lens = utilities.load_data2labels_IO(test_file)
    dev_x, dev_y, dev_lens = utilities.load_data2labels_IO(dev_file)
    label2str = utilities.IO_label2str
    selected_modules=None
    num_classes = 5

train_sents = utilities.data2sents(train_x, train_y)
test_sents = utilities.data2sents(test_x, test_y)
dev_sents = utilities.data2sents(dev_x, dev_y)
logger.info("Training size: {}".format(len(train_sents)))

# build vocabulary
logger.info("Max document length:".format(max_len))
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

embedding_size = np.shape(w2v)[1]
embedding_table = np.asarray(w2v)
agent=ALAgent(max_len, embedding_size, vocab_size, embedding_table, num_classes, policy_output)
agent.load_model(policy_path, selected_modules=selected_modules)
agent.update_embeddings(embedding_table)
start_time = time.time()
allf1list=[]  

#Training the policy
for tau in range(0, args.timesteps):
    f1list=[]
    #Shuffle train_sents, split into train_la and train_pool
    logger.info('Repetition:' + str(tau + 1))
    tagger_output = "{}/{}_tagger_r_{}.h5".format(args.output, DATASET_NAME, tau)
    logger.info(" * TAGGER OUTPUT file: {}".format(tagger_output))
    indices = np.arange(len(train_sents))
    np.random.shuffle(indices)
    train_la=[]
    train_pool=[]
    train_la_idx=[]
    train_pool_idx=[]
    for i in range(0,len(train_sents)):
        if(i<args.initial_training_size):
            train_la.append(train_sents[indices[i]])
            train_la_idx.append(train_idx[indices[i]])
        else:
            train_pool.append(train_sents[indices[i]])
            train_pool_idx.append(train_idx[indices[i]])
    #Initialise the model
    if args.model_path is not None:
        copyfile(args.model_path, tagger_output)
    model = CRFTagger(tagger_output, num_classes=num_classes)
    if args.initial_training_size > 0:
        model.train(train_la)

    coin=np.random.rand(1)
    states=[]
    actions=[]
    for t in range(0, BUDGET):
        logger.info(' * Repetition: {} Budget so far: {}'.format(str(tau+1), str(t+1)))
        '''
        queryscores=[]
        states=[]
        for i in range(0, int(len(train_pool)/k_num)):
            temp_pool=train_pool[(k_num*i):(k_num*i+k_num)]
            temp_pool_idx=train_pool_idx[(k_num*i):(k_num*i+k_num)]
            tempstate=utilities.getAllState(temp_pool,temp_pool_idx, model, w2v, 200)
            states.append(tempstate)
        states=np.array(states)
        allscores=utilities.get_intermediatelayer(policy, 2, states)[0]
        queryscores=list(itertools.chain.from_iterable(allscores.tolist()))
            #print(tempscores)
        theindex=queryscores.index(max(queryscores))
        print(theindex)
        train_la.append(train_pool[theindex])
        train_la_idx.append(train_pool_idx[theindex])
        model.train(train_la)
        #delete the selected data point from the pool
        
        del train_pool[theindex]
        del train_pool_idx[theindex]
        '''
        #Random get k sample from train_pool and train_pool_idx
        if args.al_candidate_selection_mode == 'random':
            logger.info(" * Random candidate selections")
            random_pool, random_pool_idx, queryindices = utilities.randomKSamples(train_pool, train_pool_idx, args.k_learning)
        elif args.al_candidate_selection_mode == 'uncertainty':
            logger.info(" * Uncertainty candidate selections")
            random_pool, random_pool_idx, queryindices = utilities.sample_from_top_n_uncertainty(train_pool, train_pool_idx,
                                                                                               model, args.n_learning,
                                                                                               args.k_learning)
        else:
            logger.info(" * Unknown mode. Use Random candidate selections")
            random_pool, random_pool_idx, queryindices = utilities.randomKSamples(train_pool, train_pool_idx,
                                                                                  args.k_learning)
        #get the state and action
        state=utilities.getAllState(train_la_idx, random_pool,random_pool_idx, model, w2v, max_len, num_classes)
        action=agent.predict(args.k_learning, state)
        # logger.info(action)
        
        theindex=queryindices[action]
        
        train_la.append(train_pool[theindex])
        train_la_idx.append(train_pool_idx[theindex])
        model.train(train_la)

        #delete the selected data point from the pool
        del train_pool[theindex]
        del train_pool_idx[theindex]  
        
        if((t+1) % 5 == 0):
            f1score=model.test(test_sents, label2str)
            f1list.append(f1score)
            logger.info('[Learning phase] Budget used so far: {}'.format(str(t)))
            logger.info(' * Labeled data size: {}'.format(str(len(train_la))))
            logger.info(" [Step {}] F1 score : {}".format(str(t), str(f1score)))
    allf1list.append(f1list)

    f1array=np.array(allf1list)
    averageacc=list(np.mean(f1array, axis=0))
    print('F1 list: ')
    print(allf1list)
    ww=open(resultname,'w')
    ww.writelines(str(line)+ "\n" for line in averageacc)
    ww.close()
    print("Test:--- %s seconds ---" % (time.time() - start_time))

logger.info(resultname)