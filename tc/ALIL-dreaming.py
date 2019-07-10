# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:03:05 2017

@author: lming
"""
import gc
import time

import utils
from model import *
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from queryStrategy import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

start_time = time.time()
args = utils.get_args()
logger = utils.init_logger()

EMBEDDING_DIM = args.embedding_dim
MAX_SEQUENCE_LENGTH = args.max_seq_length
MAX_NB_WORDS = args.max_nb_words

rootdir = args.root_dir
DATASET_NAME = args.dataset_name
TEXT_DATA_DIR = args.text_data_dir
TEST_DIR = args.test_set
GLOVE_DIR = args.word_vec_dir

QUERY = args.query_strategy
timesteps = args.timesteps
k_num = args.k
BUDGET = args.annotation_budget
DREAM_BUDGET = args.dreaming_budget
state_dim = 104

policyname = args.policy_path
resultname = "{}/{}_accuracy.txt".format(args.output, DATASET_NAME)

logger.info("Transfer AL policy [{}] to task on dataset {}".format(QUERY, DATASET_NAME))
logger.info(" * POLICY path: {}".format(policyname))
logger.info(" * INPUT directory: {}".format(TEXT_DATA_DIR))
logger.info(" * OUTPUT file: {}".format(resultname))

if not policyname:
    logger.info("Missing pretrained AL policy path. Use cold policy")

# first, build index mapping words in the embeddings set
# to their embedding vector
embeddings_index = utils.load_embeddings(GLOVE_DIR)

# second, prepare text samples and their labels
data, labels, word_index = utils.load_data(TEXT_DATA_DIR, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
# test_data, test_labels, _ = utils.load_data(TEST_DIR, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
# test_data, test_labels = utils.shuffle_test_data(test_data, test_labels)
# data set for inisialize the model

x_un, y_un = data, labels
embedding_matrix, num_words = utils.construct_embedding_table(embeddings_index, word_index, MAX_NB_WORDS, EMBEDDING_DIM)

def learning_phase(x_trn, y_trn, x_pool, y_pool, x_val, y_val, x_test, y_test, model, policy, max_data_points, step, accuracy_list):
    logger.info(' * Start learning phase *')
    for t in range(0, max_data_points):
        step += 1

        x_rand_unl, y_rand_unl, queryindices = sample_from_top_n_uncertainty(x_pool, y_pool,
                                                                                            model, args.n_learning,
                                                                                            args.k_learning)
        # Use the policy to get best sample
        state = getAState(x_trn, y_trn, x_rand_unl, model)
        tempstates = np.expand_dims(state, axis=0)
        action = policy.predict_classes(tempstates, verbose=0)[0]
        x_new = x_rand_unl[action]
        y_new = y_rand_unl[action]

        x_trn = np.vstack([x_trn, x_new])
        y_trn = np.vstack([y_trn, y_new])
        model.fit(x_trn, y_trn, validation_data=(x_val, y_val),
                  batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)

        index_new = queryindices[action]
        del x_pool[index_new]
        del y_pool[index_new]

        if (step +1) % 5 == 0:
            loss, mse, acc = model.evaluate(x_test, y_test, verbose=0)
            accuracy_list.append(acc)
            logger.info('[Learning phase] Budget used so far: {}'.format(str(step)))
            logger.info(' * Labeled data size: {}'.format(str(len(x_trn))))
            logger.info(" [Step {}] Accurary : {}".format(str(step), str(acc)))
    return model, step, accuracy_list, x_trn, y_trn, x_pool, y_pool

def dreaming_phase(dx_la, dy_la, x_un, y_un, budget, episodes, policy_path, expert_path):
    logger.info(' * Start Dreaming phase * ')
    states = []
    actions = []
    for tau in range(0, episodes):
        # Shuffle and split initial train,  validation set
        indices = np.arange(len(dx_la))
        np.random.shuffle(indices)
        x_la = dx_la[indices]
        y_la = dy_la[indices]
        dx_trn = x_la[:args.initial_training_size]
        dy_trn = y_la[:args.initial_training_size]

        dx_val = x_la[args.initial_training_size:]
        # dx_val = dx_val[:args.validation_size]
        dy_val = y_la[args.initial_training_size:]
        # dy_val = dy_val[:args.validation_size]

        dx_pool = list(x_un)
        dy_pool = list(y_un)
        logger.info("[Episode {}] Partition data: labeled = {}, val = {}, unlabeled pool = {} ".format(str(tau), len(dx_trn), len(dx_val), len(dx_pool)))

        logger.info("[Episode {}] Load Policy from path {}".format(str(tau), policy_path))
        policy = load_model(policy_path)
        logger.info("[Episode {}] Load Expert model from path {}".format(str(tau), expert_path))
        expert = load_model(expert_path)

        # Initilize classifier
        model = getClassifier(num_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, model_path=args.model_path)
        initial_weights = model.get_weights()
        model.fit(dx_trn, dy_trn, validation_data=(dx_val, dy_val), batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
        current_weights = model.get_weights()

        # Memory (two lists) to store states and actions
        # In every episode, run the trajectory
        for t in range(0, budget):
            logger.info('[Dreaming phase] Episode:' + str(tau + 1) + ' Budget:' + str(t + 1))
            accuracy = -1
            row = 0
            # save the index of best data point or acturally the index of action
            bestindex = 0
            # Random sample k points from D_pool
            if args.dreaming_candidate_selection_mode  == 'random':
                logger.info(" * Random candidate selections")
                x_rand_unl, y_rand_unl, queryindices = randomKSamples(dx_pool, dy_pool, k_num)
            elif args.dreaming_candidate_selection_mode  == 'certainty':
                logger.info(" * Certainty candidate selections")
                x_rand_unl, y_rand_unl, queryindices = sample_from_top_n_certainty(dx_pool, dy_pool,
                                                                                                   expert, args.n_learning, k_num)
            elif args.dreaming_candidate_selection_mode  == 'mix':
                logger.info(" * Mix method candidate selections")
                c = np.random.rand(1)
                if c > 0.5:
                    x_rand_unl, y_rand_unl, queryindices = randomKSamples(dx_pool, dy_pool, k_num)
                else:
                    x_rand_unl, y_rand_unl, queryindices = sample_from_top_n_certainty(dx_pool, dy_pool,
                                                                                                       expert,
                                                                                                       args.n_learning,
                                                                                                       k_num)
            else:
                logger.info(" * Unknown mode, use Random candidate selections")
                x_rand_unl, y_rand_unl, queryindices = randomKSamples(dx_pool, dy_pool, k_num)

            logger.debug(' * Generate label using expert')
            y_prob = expert.predict([x_rand_unl])
            y_rand_unl = to_categorical(y_prob.argmax(axis=-1), num_classes=2)

            if len(x_rand_unl) == 0:
                logger.info(" *** WARNING: Empty samples")

            for x_temp, y_temp in zip(x_rand_unl, y_rand_unl):
                model.set_weights(initial_weights)
                model.fit(dx_trn, dy_trn, batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
                x_temp_trn = np.expand_dims(x_temp, axis=0)
                y_temp_trn = np.expand_dims(y_temp, axis=0)
                history = model.fit(x_temp_trn, y_temp_trn, validation_data=(dx_val, dy_val),
                                    batch_size=args.classifier_batch_size, epochs=args.classifier_epochs,
                                         verbose=0)
                val_accuracy = history.history['val_acc'][0]
                if (val_accuracy > accuracy):
                    bestindex = row
                    accuracy = val_accuracy
                row = row + 1
            model.set_weights(current_weights)
            state = getAState(dx_trn, dy_trn, x_rand_unl, model)

            # toss a coint
            coin = np.random.rand(1)
            # if head(>0.5), use the policy; else tail(<=0.5), use the expert
            if (coin > 0.5):
                logger.debug(' * Use the POLICY [coin = {}]'.format(str(coin)))
                # tempstates= np.ndarray((1,K,len(state[0])), buffer=np.array(state))
                tempstates = np.expand_dims(state, axis=0)
                action = policy.predict_classes(tempstates)[0]
            else:
                logger.debug(' * Use the EXPERT [coin = {}]'.format(str(coin)))
                action = bestindex
            states.append(state)
            actions.append(action)
            index_new = queryindices[action]
            dx_trn = np.vstack([dx_trn, x_rand_unl[action]])
            dy_trn = np.vstack([dy_trn, y_rand_unl[action]])
            model.fit(dx_trn, dy_trn, validation_data=(dx_val, dy_val), batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
            current_weights = model.get_weights()
            del dx_pool[index_new]
            del dy_pool[index_new]

        cur_states = np.array(states)
        cur_actions = to_categorical(np.asarray(actions), num_classes=k_num)
        train_his = policy.fit(cur_states, cur_actions)
        logger.info(" [Episode {}] Training policy loss = {}, acc = {}, mean_squared_error = {}".
                    format(tau, train_his.history['loss'][0], train_his.history['acc'][0], train_his.history['mean_squared_error'][0]))
        logger.info(" * End episode {}. Save policy to {}".format(str(tau), policy_path))
        policy.save(policy_path)
        #Clear session, release memory
        K.clear_session()
        del model
        gc.collect()
        del initial_weights
        del current_weights
        gc.collect()
    return policy

logger.info('Set TF configuration')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
config.log_device_placement=False
set_session(tf.Session(config=config))

allaccuracylist=[]
for tau in range(0,args.timesteps):
    policy_output = "{}/{}_policy_fold_{}.h5".format(args.output, DATASET_NAME, tau)
    classifiername = "{}/{}_en_classifier_fold_{}.h5".format(args.output, DATASET_NAME, tau)
    logger.info(" * Validation fold: {}".format(str(tau)))

    x_la, y_la, x_un, y_un = utils.partition_data(data, labels, args.label_data_size, shuffle=True)
    x_trn, y_trn, x_valtest, y_valtest = utils.partition_data(x_la, y_la, args.initial_training_size, shuffle=True)
    x_val, y_val, x_test, y_test = utils.partition_data(x_valtest, y_valtest, args.validation_size,
                                                              shuffle=True)
    x_pool = list(x_un)
    y_pool = list(y_un)
    logger.info(
        "[Repition {}] Partition data: labeled = {}, val = {}, test = {}, unlabeled pool = {} ".format(str(tau), len(x_trn),
                                                                                            len(x_val), len(x_test), len(x_pool)))

    if not policyname:
        logger.info("[Fold {}] Init cold policy".format(str(tau)))
        policy = getPolicy(args.k, state_dim)
    else:
        logger.info("[Fold {}] Load policy from {}".format(str(tau), policyname))
        policy = load_model(policyname)
    policy.save(policy_output)
    accuracy_list = []

    logger.info("Initialize classifier")
    classifier = getClassifier(num_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH,
                               model_path=args.model_path, learning_rate=args.classifier_learning_rate)
    if args.initial_training_size > 0:
        classifier.fit(x_trn, y_trn, validation_data=(x_val, y_val), batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
        accuracy = classifier.evaluate(x_test, y_test, verbose=0)[2]
        accuracy_list.append(accuracy)
        logger.info(' * Labeled data size: {}'.format(str(len(x_trn))))
        logger.info(" [Step 0] Accurary : {}".format(str(accuracy)))
    classifier.save(classifiername)

    logger.info('Begin transfering policy..')
    step = 0
    while step < BUDGET:
        logger.info(' * Load model from path {}'.format(classifiername))
        classifier = load_model(classifiername)
        logger.info(' * Load policy from path {}'.format(policy_output))
        policy = load_model(policy_output)
        classifier, step, accuracy_list, x_trn, y_trn, x_pool, y_pool = learning_phase(x_trn, y_trn, x_pool, y_pool, x_val, y_val, x_test, y_test,
                                                   classifier, policy, args.learning_phase_length,
                                                   step, accuracy_list)
        classifier.save(classifiername)
        policy = dreaming_phase(x_trn, y_trn, x_pool, y_pool, DREAM_BUDGET, args.ndream, policy_output, classifiername)

    allaccuracylist.append(accuracy_list)

    accuracyarray=np.array(allaccuracylist)
    averageacc=list(np.mean(accuracyarray, axis=0))
    ww=open(resultname,'w')
    ww.writelines(str(line)+ "\n" for line in averageacc)
    ww.close()
    logger.info("Training complete")