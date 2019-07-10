import gc
import os
from keras.utils import to_categorical

# from alagent import ALAgent
from tagger import CRFTagger
import utilities
import tensorflow as tf
import numpy as np
import time

args = utilities.get_args()
logger = utilities.init_logger()

max_len = args.max_seq_length
VOCABULARY = args.vocab_size
EPISODES = args.episodes
BUDGET = args.annotation_budget
k = args.k

rootdir = args.root_dir
train_file = args.train_file
dev_file = args.dev_file
test_file = args.test_file
emb_file = args.word_vec_file
DATASET_NAME = args.dataset_name
policy_path = args.policy_path

tagger_temp = "{}/{}_tagger_temp.h5".format(args.output, DATASET_NAME)
resultname = "{}/{}_accuracy.txt".format(args.output, DATASET_NAME)

logger.info("Dreaming AL policy on dataset {} in {} episode with budget {}".format(DATASET_NAME, EPISODES, BUDGET))
logger.info(" * Vocabulary size: {}".format(VOCABULARY))
logger.info(" * INPUT policy: {}".format(policy_path))
logger.info(" * INPUT train file: {}".format(train_file))
logger.info(" * INPUT dev file: {}".format(dev_file))
logger.info(" * INPUT test file: {}".format(test_file))
logger.info(" * ACC OUTPUT file: {}".format(resultname))


if args.ber_task:
    logger.info("Processing data for BER task")
    train_x, train_y, train_lens = utilities.load_data2labels_BER(train_file)
    test_x, test_y, test_lens = utilities.load_data2labels_BER(test_file)
    dev_x, dev_y, dev_lens = utilities.load_data2labels_BER(dev_file)
    label2str = utilities.BER_label2str
    selected_modules = ["policy_net", "marginal_prob_cnn"]
    num_classes = 6
elif args.ibo_scheme:
    logger.info("Processing data - IBO scheme")
    train_x, train_y, train_lens = utilities.load_data2labels_IBO(train_file)
    test_x, test_y, test_lens = utilities.load_data2labels_IBO(test_file)
    dev_x, dev_y, dev_lens = utilities.load_data2labels_IBO(dev_file)
    label2str = utilities.IBO_label2str
    selected_modules = ["sentence_cnn","marginal_prob_cnn","labeled_pool","policy_net"]
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
start_time = time.time()
allf1list = []

def learning_phase(sent_trn, idx_trn, sent_pool, idx_pool, sent_test, model, agent, max_data_points, step, f1_list):
    logger.info(' * Start learning phase *')
    for t in range(0, max_data_points):
        step += 1
        # Random sample k points from D_un
        sent_rand_unl, idx_rand_unl, queryindices = utilities.randomKSamples(sent_pool, idx_pool, k)

        # Use the policy to get best sample
        state = utilities.getAllState(idx_trn, sent_rand_unl, idx_rand_unl, model, w2v, max_len)
        action=agent.predict(state)[0]

        theindex = queryindices[action]

        sent_trn.append(sent_pool[theindex])
        idx_trn.append(idx_pool[theindex])
        model.train(sent_trn)

        # delete the selected data point from the pool
        del sent_pool[theindex]
        del idx_pool[theindex]

        if (step + 1) % 5 == 0:
            f1_score = model.test(sent_test, label2str)
            f1_list.append(f1_score)
            logger.info('[Learning phase] Budget used so far: {}'.format(str(step)))
            logger.info(' * Labeled data size: {}'.format(str(len(sent_trn))))
            logger.info(' * Unlabeled data size: {}'.format(str(len(sent_pool))))
            logger.info(" [Step {}] Accurary : {}".format(str(step), str(f1_score)))
    return model, step, f1_list, sent_trn, idx_trn, sent_pool, idx_pool

def dreaming_phase(train_la, train_la_idx, train_pool, train_pool_idx, sent_dev, budget, episodes, agent, expert):
    logger.info(' * Start Dreaming phase * ')
    states = []
    actions = []
    for tau in range(0, episodes):
        # Shuffle and split initial train,  validation set
        sent_trn = list(train_la)
        idx_trn = list(train_la_idx)
        sent_pool = list(train_pool)
        idx_pool = list(train_pool_idx)
        logger.info("[Episode {}] Partition data: labeled = {}, val = {}, unlabeled pool = {} ".
                    format(str(tau), len(sent_trn), len(sent_dev), len(sent_pool)))

        # Memory (two lists) to store states and actions
        tagger_dreamming = "{}/{}_tagger_temp.h5".format(args.output, DATASET_NAME)
        if os.path.exists(tagger_dreamming):
            os.remove(tagger_dreamming)
        model = CRFTagger(tagger_dreamming, num_classes=num_classes)

        if len(sent_trn) > 0:
            model.train(sent_trn)

        # In every episode, run the trajectory
        for t in range(0, budget):
            logger.info('[Dreaming phase] Episode:' + str(tau + 1) + ' Budget:' + str(t + 1))
            row = 0
            f1 = -1
            # save the index of best data point or acturally the index of action
            bestindex = 0
            # Random sample k points from D_pool
            random_pool, random_pool_idx, queryindices = utilities.randomKSamples(sent_pool, idx_pool, k)
            logger.debug(' * Generate label using expert')
            x_tokens = [' '.join(expert.sent2tokens(s)) for s in random_pool]
            y_labels = expert.predict(random_pool)
            pred_sents = utilities.data2sents(x_tokens, y_labels)

            for datapoint in zip(pred_sents, random_pool_idx):
                seq = datapoint[0]
                idx = datapoint[1]
                train_la_temp = list(sent_trn)
                train_la_temp.append(seq)

                if os.path.exists(tagger_temp):
                    os.remove(tagger_temp)
                model_temp = CRFTagger(tagger_temp, num_classes=num_classes)
                model_temp.train(train_la_temp)
                f1_temp = model_temp.test(dev_sents, label2str)
                if (f1_temp > f1):
                    bestindex = row
                    f1 = f1_temp
                row = row + 1
                del model_temp
                del train_la_temp
                gc.collect()

            # get the state and action
            state = utilities.getAllState(idx_trn, random_pool, random_pool_idx, model, w2v, max_len)
            # action=bestindex
            coin = np.random.rand(1)
            if (coin > 0.5):
                logger.debug(' * Use the POLICY [coin = {}]'.format(str(coin)))
                action = bestindex
            else:
                action=agent.predict(state)[0]
            states.append(state)
            actions.append(action)
            # update the model

            theindex = queryindices[bestindex]
            sent_trn.append(sent_pool[theindex])
            idx_trn.append(idx_pool[theindex])

            model.train(sent_trn)
            # delete the selected data point from the pool
            del sent_pool[theindex]
            del idx_pool[theindex]

        cur_actions = to_categorical(np.asarray(actions), num_classes=k)
        agent.train_policy(states, cur_actions)
        del sent_pool
        del idx_pool
        del sent_trn
        del idx_trn
        gc.collect()
    return agent

# start_time = time.time()
# allf1list=[]
#
# agent=ALAgent(k, max_len, embedding_size, vocab_size, embedding_table, num_classes, None)
# agent.load_model(policy_path, selected_modules=selected_modules)
# agent.update_embeddings(embedding_table)
#
#
# for r in range(0, args.timesteps):
#     logger.info('Repetition:' + str(r + 1))
#     policy_output = "{}/{}_policy_r_{}.ckpt".format(args.output, DATASET_NAME, r)
#     tagger_output = "{}/{}_tagger_r_{}.h5".format(args.output, DATASET_NAME, r)
#     logger.info(" * POLICY OUTPUT path: {}".format(policy_output))
#     logger.info(" * TAGGER OUTPUT file: {}".format(tagger_output))
#     logger.info("[Repetition {}] Load policy from {}".format(str(r), policy_path))
#
#     logger.info(" * POLICY OUTPUT path: {}".format(policy_output))
#     logger.info(" * TAGGER OUTPUT file: {}".format(tagger_output))
#     agent.load_model(policy_path, selected_modules=selected_modules, black_list=black_list)
#     agent.update_embeddings(embedding_table)
#     agent.policy_output = policy_output
#     agent.save_model()
#
#     logger.info("[Repetition {}] Parition training data to labeled and unlabeled data".format(r+1))
#     indices = np.arange(len(train_sents))
#     np.random.shuffle(indices)
#     train_la = []
#     train_pool = []
#     train_la_idx = []
#     train_pool_idx = []
#     for i in range(0, len(train_sents)):
#         if (i < args.initial_training_size):
#             train_la.append(train_sents[indices[i]])
#             train_la_idx.append(train_idx[indices[i]])
#         else:
#             train_pool.append(train_sents[indices[i]])
#             train_pool_idx.append(train_idx[indices[i]])
#
#
#     logger.info(' * Begin dreaming policy..')
#     step = 0
#     f1_list = []
#     tagger = CRFTagger(tagger_output, num_classes=num_classes)
#     if args.initial_training_size > 0:
#         tagger.train(train_la)
#     while step < BUDGET:
#         tagger, step, f1_list, train_la, train_la_idx, train_pool, train_pool_idx = learning_phase(train_la, train_la_idx, train_pool
#                                                                                            , train_pool_idx, test_sents,
#                                                    tagger, agent, args.learning_phase_length,
#                                                    step, f1_list)
#         agent = dreaming_phase(train_la, train_la_idx, train_pool, train_pool_idx, dev_sents, args.dreaming_budget,
#                                 args.ndream, agent, tagger)
#         logger.info("Save policy to {}".format(policy_output))
#         agent.save_model()
#
#     allf1list.append(f1_list)
#
#     f1array=np.array(allf1list)
#     averageacc=list(np.mean(f1array, axis=0))
#     print('F1 list: ')
#     print(allf1list)
#     ww=open(resultname,'w')
#     ww.writelines(str(line)+ "\n" for line in averageacc)
#     ww.close()
#     print("Test:--- %s seconds ---" % (time.time() - start_time))
#
# logger.info(resultname)
logger.info(">>>>> Dataset {} size {}".format(DATASET_NAME, len(train_sents)))
# num_initial_data = [10, 20, 50, 100, 150, 200, 300, 400, 500, 1000]
# for n in num_initial_data:
#     f1s = []
    # for i in range(50):
    #     sent_trn, idx_trn, query = utilities.randomKSamples(train_sents, train_idx, n)
tagger_output = "{}/{}_tagger.h5".format(args.output, DATASET_NAME)
if os.path.exists(tagger_output):
    os.remove(tagger_output)
model = CRFTagger(tagger_output, num_classes=num_classes)
model.train(train_sents)
f1_score = model.test(test_sents, label2str)
logger.info('*********************************')
logger.info(' F1 score : {}'.format(f1_score))
    # f1_max = np.max(np.asarray(f1s))
    # f1_mean = np.mean(np.asarray(f1s))
    # f1_min = np.min(np.asarray(f1s))
    # logger.info(' * Average f1 scrore: {}'.format(f1_mean))
    # logger.info(' * Min f1 scrore: {}'.format(f1_min))
    # logger.info(' * Max f1 scrore: {}'.format(f1_max))