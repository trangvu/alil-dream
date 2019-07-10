import argparse
import logging
import math

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

# map a label to a string
IO_label2str = {1: "PER", 2: "LOC", 3: "ORG", 4: "MISC", 5: "O"}
IBO_label2str = {1: "O", 2: "I-PER", 3: "B-PER", 4: "B-LOC", 5: "I-LOC", 6: "I-ORG", 7: "B-ORG", 8: "I-MISC", 9: "B-MISC"}
BER_label2str = {1: "DNA", 2: "cell_line", 3: "protein", 4: "cell_type", 5: "RNA", 6: "O"}

# load original data

logger = logging.getLogger()


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=40,
                        help='embedding size')
    parser.add_argument('--max_seq_length', type=int, default=120,
                        help='max sequence length')
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='max vocab size')
    parser.add_argument('--dataset_name',
                        help='dataset name')
    parser.add_argument('--root_dir',
                        help='root directory')
    parser.add_argument('--train_file',
                        help='path to train file')
    parser.add_argument('--dev_file',
                        help='path to dev file')
    parser.add_argument('--test_file', nargs='?',
                        help='path to test file')
    parser.add_argument('--word_vec_file',
                        help='word vector file')
    parser.add_argument('--query_strategy', nargs='?',
                        help='data point query strategy: Random, Uncertain, Diversity')
    parser.add_argument('--episodes', type=int,
                        help='number of training episodes')
    parser.add_argument('--ndream', type=int, default=5,
                        help='number of dream in dreaming phase')
    parser.add_argument('--timesteps', type=int,
                        help='number of training timesteps in a episode')
    parser.add_argument('--k', type=int, default=5,
                        help='k - number of samples to query each time')
    parser.add_argument('--k_learning', type=int, default=10,
                        help='k value in active learning phase')
    parser.add_argument('--n_learning', type=int, default=100,
                        help='top n uncertainty for candidate selection in active learning phase')
    parser.add_argument('--annotation_budget', type=int, default=200,
                        help='annotation budget')
    parser.add_argument('--dreaming_budget', type=int, default=10,
                        help='dreaming budget')
    parser.add_argument('--output',
                        help='Output folder')
    parser.add_argument('--validation_size', type=int, default=10,
                        help='Number of data to leave out for validation in each episode')
    parser.add_argument('--label_data_size', type=int, default=200,
                        help='Number of labeled data. The rest will be treated as unlabel data in each episode')
    parser.add_argument('--initial_training_size', type=int, default=20,
                        help='Number of data point to initialize underlying model')
    parser.add_argument('--policy_path', default="",
                        help='policy path')
    parser.add_argument('--model_path', default=None,
                        help='model path')
    parser.add_argument('--learning_phase_length', type=int, default=10,
                        help='number of datapoint to get annotation on ')
    parser.add_argument('--ibo_scheme', action='store_true',
                        help='If set, we use ibo scheme. otherwise IO scheme ')
    parser.add_argument('--ber_task', action='store_true',
                        help='If set, train CRF tagger for BER task')
    parser.add_argument('--dreaming_candidate_selection_mode', default="random",
                        help='How to select candidate for dreaming: random, certainty, mix')
    parser.add_argument('--al_candidate_selection_mode', default="uncertainty",
                        help='How to select candidate in AL phase: random, uncertainty')
    parser.add_argument('--dream_increase_step', type=int, default=1000,
                        help='Increase dream rate')

    return parser.parse_args()

def load_data2labels_IO(input_file):
    labels_map = {'B-ORG': 3, 'O': 5, 'B-MISC': 4, 'B-PER': 1,
                  'I-PER': 1, 'B-LOC': 2, 'I-ORG': 3, 'I-MISC': 4, 'I-LOC': 2}
    return load_data2labels(input_file, labels_map)

def load_data2labels_IBO(input_file):
    labels_map = {'B-ORG': 7, 'O': 1, 'B-MISC': 9, 'B-PER': 3,
                  'I-PER': 2, 'B-LOC': 4, 'I-ORG': 6, 'I-MISC': 8, 'I-LOC': 5}
    return load_data2labels(input_file, labels_map)

def load_data2labels_BER(input_file):
    #Label maps for Bio Entity Recognition
    labels_map = {'B-protein': 3, 'O': 6, 'B-cell_type': 4, 'B-DNA': 1, 'B-RNA': 5, 'I-RNA': 5,
                  'I-DNA': 1, 'B-cell_line': 2, 'I-protein': 3, 'I-cell_type': 4, 'I-cell_line': 2}
    return load_data2labels(input_file, labels_map)

def load_data2labels(input_file, labels_map):
    # predefine a label_set: PER - 1 LOC - 2 ORG - 3 MISC - 4 O - 5
    # 0 is for padding
    seq_set = []
    seq = []
    seq_set_label = []
    seq_label = []
    seq_set_len = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                seq_set.append(" ".join(seq))
                seq_set_label.append(seq_label)
                seq_set_len.append(len(seq_label))
                seq = []
                seq_label = []
            else:
                tok, label = line.split()
                seq.append(tok)
                seq_label.append(labels_map[label])
    return [seq_set, seq_set_label, seq_set_len]


def load_crosslingual_embeddings(inputFile, vocab, max_vocab_size=20000):
    embeddings = list(open(inputFile, "r").readlines())
    pre_w2v = {}
    emb_size = 0
    for emb in embeddings:
        parts = emb.strip().split()
        if emb_size != (len(parts) - 1):
            if emb_size == 0:
                emb_size = len(parts) - 1
            else:
                logger.info("Different embedding size!")
                break

        w = parts[0]
        w_parts = w.split(":")
        if len(w_parts) != 2:
            w = ":"
        else:
            w = w_parts[1]
        vals = []
        for i in range(1, len(parts)):
            vals.append(float(parts[i]))
        # print w, vals
        pre_w2v[w] = vals

    n_dict = len(vocab._mapping)
    vocab_w2v = [None] * n_dict
    # vocab_w2v[0]=np.random.uniform(-0.25,0.25,100)
    for w, i in vocab._mapping.items():
        if w in pre_w2v:
            vocab_w2v[i] = pre_w2v[w]
        else:
            vocab_w2v[i] = list(np.random.uniform(-0.25, 0.25, emb_size))

    cur_i = len(vocab_w2v)
    if len(vocab_w2v) > max_vocab_size:
        logger.info("Vocabulary size is larger than {}".format(max_vocab_size))
        raise SystemExit
    while cur_i < max_vocab_size:
        cur_i += 1
        padding = [0] * emb_size
        vocab_w2v.append(padding)
    logger.info("Vocabulary {} Embedding size {}".format(n_dict, emb_size))
    return vocab_w2v
#return a matrix, in which each row id is the word id, the row is the embedding

def data2sents(X, Y):
    data = []
    for i in range(len(Y)):
        sent = []
        text = X[i]
        items = text.split()
        for j in range(len(Y[i])):
            sent.append((items[j], str(Y[i][j])))
        #elements in data is tuple (token,label)
        data.append(sent)
    return data

def sents2Xdata(sents):
    data=[]
    for item in sents:
        text=''
        for point in item:
            text=text+' '+point[0]
        data.append(text)
    return data

def randomKSamples(train_pool, train_pool_idx, num):
    #x_un=np.array(x_un)
    #y_un=np.array(y_un)
    random_pool=[]
    random_pool_idx=[]
    indices=np.arange(len(train_pool))
    np.random.shuffle(indices)
    queryindices=indices[0:num]
    for i in range(0,num):
        random_pool.append(train_pool[indices[i]])
        random_pool_idx.append(train_pool_idx[indices[i]])
    return random_pool, random_pool_idx, queryindices

def getCRFunRank(train_pool, model, data_new):
    rank=1
    entropylist=[]
    for x in train_pool:
        confidence=model.get_confidence(x)
        entropylist.append(getEntropy(confidence))
    data_entropy=getEntropy(model.get_confidence(data_new))
    sorted_entropy=sorted(entropylist,reverse=True)
    for i in range(0,len(sorted_entropy)):
        if((data_entropy-sorted_entropy[i])>0):
            rank=i+1
            break
        else:
            continue
    del entropylist
    del sorted_entropy
    del data_entropy
    return rank

def getCRFdiRand(train_la_idx, train_pool_idx, model, x_new_idx):
    rank=1
    diversity=[]
    results_union = set().union(*train_la_idx)
    x_new_similarity=jaccard_similarity(x_new_idx, results_union)
    for example in train_pool_idx:
        value=jaccard_similarity(example, results_union)
        diversity.append(value)
    sorted_diversity=sorted(diversity, reverse=True)
    for i in range(0,len(sorted_diversity)):
        if((x_new_similarity-sorted_diversity[i])>0):
            rank=i+1
            break
        else:
            continue
    del diversity
    del sorted_diversity
    return rank

def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality) 

def getEntropy(v):
    entropy=0.
    for element in v:
        p=float(element)
        if p > 0:
            entropy=entropy+ (-p)* np.log(p)
    return entropy

def compute_entropy(y, num_class):
    ent = 0.
    for y_i in y:
        if y_i > 0:
            ent -= y_i * math.log(y_i, num_class)
    return ent
def get_top_uncertainty(train_sent, train_idx, model, num):
    sent = np.array(train_sent)
    idx = np.array(train_idx)
    entropy = []
    entropy = [model.get_uncertainty(item) for item in sent]
    indices = sorted(range(len(entropy)), key=lambda i: entropy[i], reverse=True)
    top_data = []
    top_idx = []
    for i in range(0, num):
        top_data.append(sent[indices[i]])
        top_idx.append(idx[indices[i]])
    queryindices = indices[0:num]
    return top_data, top_idx, queryindices

def get_top_certainty(train_sent, train_idx, model, num):
    sent = np.array(train_sent)
    idx = np.array(train_idx)
    entropy = []
    entropy = [model.get_uncertainty(item) for item in sent]
    indices = sorted(range(len(entropy)), key=lambda i: entropy[i], reverse=False)
    top_data = []
    top_idx = []
    for i in range(0, num):
        top_data.append(sent[indices[i]])
        top_idx.append(idx[indices[i]])
    queryindices = indices[0:num]
    return top_data, top_idx, queryindices

def sample_from_top_n_uncertainty(train_sent, train_idx, model, n, k):
    if k > n:
        raise ("n should be larger than k. Found n = "+ str(n)+ ", k=" + str(k))
    top_data, top_idx, top_query_indices = get_top_uncertainty(train_sent, train_idx, model, n)
    sample_data, sample_idx, sample_query_indices = randomKSamples(top_data, top_idx, k)
    query_indices = []
    for i in range(len(sample_query_indices)):
        query_indices.append(top_query_indices[sample_query_indices[i]])
    return sample_data, sample_idx, query_indices

def sample_from_top_n_certainty(train_sent, train_idx, model, n, k):
    if k > n:
        raise ("n should be larger than k. Found n = "+ str(n)+ ", k=" + str(k))
    top_data, top_idx, top_query_indices = get_top_certainty(train_sent, train_idx, model, n)
    sample_data, sample_idx, sample_query_indices = randomKSamples(top_data, top_idx, k)
    query_indices = []
    for i in range(len(sample_query_indices)):
        query_indices.append(top_query_indices[sample_query_indices[i]])
    return sample_data, sample_idx, query_indices

def getAllState(train_pool_idx, random_pool,random_pool_idx,model,w2v, max_len, num_class):
    lab_data_rep = [0 for x in range(len(w2v[0]))]
    for i in range(len(train_pool_idx)):
        for j in range(len(train_pool_idx[i])):
            lab_data_rep = list(map(lambda x, y: x + y, lab_data_rep, w2v[train_pool_idx[i][j]]))
    sequence_input = random_pool_idx
    predictions = [model.get_predictions(item) for item in random_pool]
    entropy = [[compute_entropy(word, num_class) for word in sent ] for sent in predictions ]
    def get_mean(sent):
        if len(sent) == 0:
            logger.error("Mean Empty sent {}".format(sent))
            return 0
        else:
            return np.mean(sent)
    def get_sum(sent):
        if len(sent) == 0:
            logger.error("Sum Empty sent {}".format(sent))
            return 0
        else:
            return np.sum(sent)
    def get_max(sent):
        if len(sent) == 0:
            logger.error("Max Empty sent {}".format(sent))
            return 0
        else:
            return np.max(sent)
    ent_stat = [[get_mean(sent), get_max(sent), get_sum(sent)] for sent in entropy]
    predictions_pad = pad_sequences(predictions, maxlen=max_len, dtype='float32', padding='post', truncating='post')
    entropy_pad = pad_sequences(entropy, maxlen=max_len, dtype='float32', padding='post', truncating='post')
    confidence = [model.get_confidence(item) for item in random_pool]
    lab_data_rep_input = [lab_data_rep for item in random_pool]
    return [np.asarray(sequence_input), np.asarray(predictions_pad), np.asarray(confidence), np.asarray(lab_data_rep_input),
            entropy_pad, ent_stat]

def get_intermediatelayer(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations

'''
train_x, train_y, train_lens =load_data2labels('C:/Users/lming/My Projects/MIME/Tools/ner/data/esp.train')
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_document_length=120, min_frequency=1)
    # vocab = vocab_processor.vocabulary_ # start from {"<UNK>":0}
train_idx = np.array(list(vocab_processor.fit_transform(train_x)))
vocab = vocab_processor.vocabulary_
print(vocab._mapping)
example=load_crosslingual_embeddings('C:/Users/lming/My Projects/MIME/Tools/ner/en-da.normalized',vocab,20000)

print(example)
'''