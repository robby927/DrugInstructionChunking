import sys, pickle, os, random
from pathlib import Path

import numpy as np

## tags, BIEO
tag2label = {"O": 0,
             "B-给药途径": 1, "I-给药途径": 2, "E-给药途径": 3,
             "B-给药频次": 4, "I-给药频次": 5, "E-给药频次": 6,
             "B-单次剂量": 7, "I-单次剂量": 8, "E-单次剂量": 9,
             "B-慎用禁忌": 10, "I-慎用禁忌": 11, "E-慎用禁忌": 12,
             "B-单日剂量": 13, "I-单日剂量": 14, "E-单日剂量": 15,
             "B-年龄": 16, "I-年龄": 17, "E-年龄": 18,
             "B-诊断": 19, "I-诊断": 20, "E-诊断": 21,
             "B-特殊人群": 22, "I-特殊人群": 23, "E-特殊人群": 24,
             "B-疗程": 25, "I-疗程": 26, "E-疗程": 27,
             "B-给药速度": 28, "I-给药速度": 29, "E-给药速度": 30,
             "B-体重": 31, "I-体重": 32, "E-体重": 33,
             "B-性别": 34, "I-性别": 35, "E-性别": 36
             }

# tag2label = {"O": 0,
#              "B-给药途径": 1, "E-给药途径": 2,
#              "B-给药频次": 3, "E-给药频次": 4,
#              "B-单次剂量": 5, "E-单次剂量": 6,
#              "B-慎用禁忌": 7, "E-慎用禁忌": 8,
#              "B-单日剂量": 9, "E-单日剂量": 10,
#              "B-年龄": 11, "E-年龄": 12,
#              "B-诊断": 13, "E-诊断": 14,
#              "B-特殊人群": 15, "E-特殊人群": 16,
#              "B-疗程": 17, "E-疗程": 18,
#              "B-给药速度": 19, "E-给药速度": 20,
#              "B-体重": 21, "E-体重": 22,
#              "B-性别": 23, "E-性别": 24
#              }

# tag2label = {"O": 0, "B": 1, "I": 2 }


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    idx = 0
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_, pos_, ner_ = [], [], [], []

    for line in lines:
        idx += 1
        if line.find("DOC-ID") < 0 and line != '\n':
            try:
                [char, label, pos_tag, ner_tag] = line.strip().split()
                sent_.append(char)
                tag_.append(label)
                pos_.append(pos_tag)
                ner_.append(ner_tag)
            except:
                print(line)
        else:
            # print(line)
            if idx > 1:
                data.append((sent_, tag_, pos_, ner_))
                sent_, tag_, pos_, ner_ = [], [], [], []

    return data

def read_corpus_2(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    idx = 0
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []

    for line in lines:
        idx += 1
        if line.find("DOC-ID") < 0 and line != '\n':
            try:
                [char, label] = line.strip().split()
                sent_.append(char)
                tag_.append(label)
            except:
                print(line)
        else:
            # print(line)
            if idx > 1:
                data.append((sent_, tag_))
                sent_, tag_ = [], []

    return data

def read_corpus_3(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    idx = 0
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_h_, tag_p_ = [], [], []

    for line in lines:
        idx += 1
        if line.find("DOC-ID") < 0 and line != '\n':
            try:
                [char, tag_h, tag_p] = line.strip().split()
                sent_.append(char)
                tag_h_.append(tag_h)
                tag_p_.append(tag_p)
            except:
                print(line)
        else:
            # print(line)
            if idx > 1:
                data.append((sent_, tag_h_, tag_p_))
                sent_, tag_h_, tag_p_ = [], [], []

    return data

def read_tuples_list(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    idx = 0
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, label_h_, pos_, ner_ = [], [], [], []

    for line in lines:
        idx += 1
        if line.find("DOC-ID") < 0 and line != '\n':
            try:
                [char, label_h, pos_tag, ner_tag] = line.strip().split()
                sent_.append(char)
                label_h_.append(label_h)
                pos_.append(pos_tag)
                ner_.append(ner_tag)
            except BaseException as e:
                print(e)
                print(line)
        else:
            # print(line)
            if idx > 1:
                data.append((sent_, label_h_, pos_, ner_))
                sent_, label_h_, pos_, ner_ = [], [], [], []

    return data

def vocab_build(vocab_path, train_corpus_path, test_corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    vocab_words = []
    train_data = read_corpus(train_corpus_path)
    test_data = read_corpus(test_corpus_path)
    data = train_data + test_data
    word2id = {}
    for sent_, tag_, pos, ner in data:
        for word in sent_:
            # if word.isdigit():
            #     word = '<NUM>'
            # elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
            #     word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        vocab_words.append(word)
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0
    vocab_words.append('<UNK>')
    vocab_words.append('<PAD>')

    print('construct word2id dict, size = {}', len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)

    print('words size : ', len(vocab_words))
    with open('data_path/vocab.words.txt', 'w+') as f:
        for w in vocab_words:
            f.write('{}\n'.format(w))


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
