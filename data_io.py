import sys, pickle, os, random
from pathlib import Path

import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-给药途径": 1, "I-给药途径": 2,
             "B-给药频次": 3, "I-给药频次": 4,
             "B-单次剂量": 5, "I-单次剂量": 6,
             "B-慎用禁忌": 7, "I-慎用禁忌": 8,
             "B-单日剂量": 9, "I-单日剂量": 10,
             "B-年龄": 11, "I-年龄": 12,
             "B-诊断": 13, "I-诊断": 14,
             "B-特殊人群": 15, "I-特殊人群": 16,
             "B-疗程": 17, "I-疗程": 18,
             "B-给药速度": 19, "I-给药速度": 20,
             "B-体重": 21, "I-体重": 22,
             "B-性别": 23, "I-性别": 24
             }

# tag2label = {"O": 0, "B": 1, "I": 2 }


def read_predicted_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    idx = 0
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, label_h_, label_p_, pos_, ner_ = [], [], [], [], []

    for line in lines:
        idx += 1
        if line.find("DOC-ID") < 0 and line != '\n':
            try:
                [char, label_h, label_p, pos_tag, ner_tag] = line.strip().split(' ')
                sent_.append(char)
                label_h_.append(label_h)
                label_p_.append(label_p)
                pos_.append(pos_tag)
                ner_.append(ner_tag)
            except BaseException as e:
                print(e)
                print(line)
        else:
            # print(line)
            if idx > 1:
                data.append((sent_, label_h_, label_p_, pos_, ner_))
                sent_, label_h_, label_p_, pos_, ner_ = [], [], [], [], []
    # data.append((sent_, label_h_, label_p_, pos_, ner_))
    return data


def read_original_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    idx = 0
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, label_h_, label_p_, pos_, ner_ = [], [], [], [], []

    for line in lines:
        idx += 1
        if line.find("DOC-ID") < 0 and line != '\n':
            try:
                [char, label_h, label_p, pos_tag, ner_tag] = line.strip().split(' ')
                sent_.append(char)
                label_h_.append(label_h)
                label_p_.append('O')
                pos_.append(pos_tag)
                ner_.append(ner_tag)
            except BaseException as e:
                print(e)
                print(line)
        else:
            # print(line)
            if idx > 1:
                data.append((sent_, label_h_, label_p_, pos_, ner_))
                sent_, label_h_, label_p_, pos_, ner_ = [], [], [], [], []

    return data

def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_, pos_, ner_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def get_pretrain_embeddings(vocab_path, embeddings_path):
    # Load vocab
    with Path(vocab_path).open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.zeros((size_vocab, 300))

    # Get relevant glove vectors
    found = 0
    print('Reading GloVe file (may take a while)')
    with Path(embeddings_path).open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))
    # Save np.array to file
    np.savez('glove.npy', embeddings=embeddings)
    return embeddings

def load_predicted_result(self, file_path):
    corpus_ = []
    sent_ = []
    for line in open(file_path, encoding='utf-8'):
        if line != '\n':
            try:
                [char, label_h, label_p] = line.strip().split(' ')
                sent_.append([char, label_h, label_p])
            except:
                print(line)
        else:
            corpus_.append(sent_)
            sent_ = []

    return corpus_

def tree_has_word(words_list, tree):
    for words in words_list:
        matched_num = 0
        for word in words:
            for (word_, pos_) in tree:
                if word == word_:
                    matched_num += 1

        if matched_num == len(words):
            return True

    return False

def write_model_predict(model_predict_pos_ner, label_path):
    """

        :param label_predict:
        :param label_path:
        :param metric_path:
        :return:
        """

    with open(label_path, "w") as fw:
        line = []
        for char_, tag_h_, tag_p_, pos_, ner_ in model_predict_pos_ner:
            for char, tag, tag_p, pos, ner in zip(char_, tag_h_, tag_p_, pos_, ner_):
                tag = '0' if tag == 'O' else tag
                line.append("{} {} {} {} {}\n".format(char, tag, tag_p, pos, ner))
            line.append("\n")
        fw.writelines(line)

def load_predicted_result(file_path):
    corpus_ = []
    sent_ = []
    for line in open(file_path, encoding='utf-8'):
        if line != '\n':
            try:
                [char, label_h, label_p, pos, ner] = line.strip().split(' ')
                sent_.append([char, label_h, label_p])
            except:
                print(line)
        else:
            corpus_.append(sent_)
            sent_ = []

    return corpus_

def load_original_result(file_path):
    corpus_ = []
    sent_ = []
    for line in open(file_path, encoding='utf-8'):
        if line != '\n':
            try:
                [char, label_h, label_p, pos, ner] = line.strip().split(' ')
                sent_.append([char, label_h, 'O'])
            except:
                print(line)
        else:
            corpus_.append(sent_)
            sent_ = []

    return corpus_