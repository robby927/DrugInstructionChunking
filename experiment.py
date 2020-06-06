import copy
from enum import Enum

from data import read_corpus, read_corpus_3
from eval import myevaluate, get_f1
from data_io import read_predicted_corpus
import numpy as np

# model_name = 'BiLSTM-CRF'
# model_name = 'BERT-BiLSTM-CRF/base'
# model_name = 'ALBERT-BiLSTM-CRF/base'


class Element:
    index = None
    name = None
    value = None
    relation = None
    extend = False

    def __init__(self, index, name, value, relation, extend=False):
        self.index = index
        self.name = name
        self.value = value
        self.relation = relation
        self.extend = extend


class Template:
    elements = None
    max_end_index = 0

    def __init__(self, elements, max_end_index):
        self.elements = elements
        self.max_end_index = max_end_index


def read_dict(filename):
    length_word_dict = dict()
    words = np.loadtxt(fname=filename, dtype=str)
    for word in words:
        length = len(word)
        if length_word_dict.get(length) is None:
            length_word_dict.update({length: []})
            for i in range(length):
                word_i = []
                length_word_dict.get(length).append(word_i)

        for i in range(length):
            length_word_dict.get(length)[i].append(word[i])
    return length_word_dict

def get_template():
    bwu_dict = read_dict('dict/body_weight_unit.dict')
    wu_dict = read_dict('dict/weight_unit.dict')
    qu_dict = read_dict('dict/quantifier.dict')
    tu_dict = read_dict('dict/time_unit.dict')
    du_dict = read_dict('dict/dosage_unit.dict')

    body_weight_units_2_1 = bwu_dict.get(2)[0]
    body_weight_units_2_2 = bwu_dict.get(2)[1]


    templates = dict()
    body_weight_templates = [
        Template([Element(-1, 'word', ['按'], '!='), Element(0, 'word', ['体'], '='),
                  Element(1, 'word', ['重'], '='), Element(5, 'ner', 'NUMBER', '=', True),
                  Element(6, 'word', body_weight_units_2_1, '='), Element(7, 'word', body_weight_units_2_2, '=')], 10),

        Template([Element(-1, 'pos', ['PU'], '='), Element(0, 'ner', ['NUMBER'], '='),
                  Element(1, 'ner', ['NUMBER'], '='), Element(2, 'pos', ['PU'], '='),
                  Element(3, 'ner', ['NUMBER'], '='), Element(4, 'ner', ['NUMBER'], '='),
                  Element(5, 'word', body_weight_units_2_1, '='), Element(6, 'word', body_weight_units_2_2, '=')], 0)
    ]
    templates.update({'体重': body_weight_templates})

    dosage_units_1_1 = du_dict.get(1)[0]
    dosage_units_2_1 = du_dict.get(2)[0]
    dosage_units_2_2 = du_dict.get(2)[1]
    time_unit_2_1 = tu_dict.get(2)[0]
    time_unit_2_2 = tu_dict.get(2)[1]
    per_words = qu_dict.get(1)[0]
    speed_templates = [
        # Template([Element(0, 'word', ['滴'], '='), Element(1, 'word', ['速'], '='), Element(2, 'ner', ['NUMBER'], '='),
        #  Element(3, 'pos', ['PU'], '='), Element(4, 'ner', ['NUMBER'], '='), Element(5, 'pos', ['PU'], '='),
        #  Element(12, 'word', time_unit_2_1, '='), Element(13, 'word', time_unit_2_2, '=')], 15),
        #
        # Template([Element(-2, 'word', ['滴'], '!='), Element(-1, 'word', ['速'], '!='), Element(0, 'ner', ['NUMBER'], '='),
        #  Element(3, 'pos', ['PU'], '='), Element(4, 'ner', ['NUMBER'], '='), Element(5, 'pos', ['PU'], '='),
        #  Element(12, 'word', time_unit_2_1, '='), Element(13, 'word', time_unit_2_2, '=')], 15),

        Template([Element(0, 'word', ['滴'], '='), Element(1, 'word', ['速'], '='),
                  Element(11, 'pos', ['PU'], '=', True),
                  Element(12, 'word', time_unit_2_1, '='), Element(13, 'word', time_unit_2_2, '=')], 17),

        Template([Element(-1, 'pos', ['PU'], '='), Element(0, 'word', per_words, '='),
                  Element(1, 'word', time_unit_2_1, '='), Element(2, 'word', time_unit_2_2, '='),
                  Element(4, 'word', '次', '!='),
                  Element(6, 'ner', ['NUMBER'], '=', True), Element(7, 'word', dosage_units_1_1, '=')], 10),

        Template([Element(-1, 'pos', ['PU'], '='), Element(0, 'word', per_words, '='),
                  Element(1, 'word', time_unit_2_1, '='), Element(2, 'word', time_unit_2_2, '='),
                  Element(4, 'word', dosage_units_2_1, '=', True), Element(5, 'word', dosage_units_2_2, '=')], 10),

        # 滴注速度
        # 注射速度
        # 0
        Template([Element(0, 'word', ['滴', '注'], '='), Element(1, 'word', ['注', '射'], '='),
                  Element(2, 'word', ['速'], '='), Element(3, 'word', ['度'], '='),
                  Element(4, 'word', per_words, '='), Element(5, 'word', time_unit_2_1, '='),
                  Element(6, 'word', time_unit_2_2, '='), Element(7, 'word', dosage_units_2_1, '=', True),
                  Element(8, 'word', dosage_units_2_2, '=')], 15),

        Template([Element(-1, 'word', ['速', '度'], '!='), Element(0, 'word', per_words, '='),
                  Element(1, 'word', time_unit_2_1, '='), Element(2, 'word', time_unit_2_2, '='),
                  Element(7, 'word', dosage_units_2_1, '=', True), Element(8, 'word', dosage_units_2_2, '=')], 15),

    ]
    templates.update({'给药速度': speed_templates})

    gender_templates = [
        Template([Element(0, 'word', ['男'], '='), Element(1, 'word', ['性'], '=')], 2),
        Template([Element(0, 'word', ['女'], '='), Element(1, 'word', ['性'], '=')], 2),
    ]
    templates.update({'性别': gender_templates})

    return templates


def data2word_pos_tag_tuple(train_data):
    return [
        [[{'word': word, 'pos': pos, 'iob': tag_h, 'tag_p': tag_p, 'ner': ner_tag}, tag_p] for
         (word, tag_h, tag_p, pos, ner_tag) in
         zip(one_sent_data[0], one_sent_data[1], one_sent_data[2], one_sent_data[3], one_sent_data[4])] for
        one_sent_data in
        train_data]


def data2eval_data(train_data):
    return [
        [(word, tag_h, tag_p) for (word, tag_h, tag_p, pos, ner_tag) in
         zip(one_sent_data[0], one_sent_data[1], one_sent_data[2], one_sent_data[3], one_sent_data[4])] for
        one_sent_data in
        train_data]


def word_pos_tag_tuple2eval_data(tuples):
    return [
        [(one_sent_tagged[0]['word'], one_sent_tagged[0]['iob'], one_sent_tagged[1]) for one_sent_tagged in taggedtuple]
        for taggedtuple in tuples]


def train():
    models = ['BiLSTM-CRF', 'ALBERT-BiLSTM-CRF/base', 'ALBERT-BiLSTM-CRF/tiny', 'BERT-BiLSTM-CRF/base']
    threshold = 0
    template_selected = dict()
    for model_name in models:
        print('\n------------------------------------' + model_name + '-------------------------------------\n')
        base_path = './dataset/' + model_name + '/'

        train_file_path = base_path + 'train_data'
        dev_file_path = base_path + 'label_dev_pos_ner'
        test_file_path = base_path + 'label_test_pos_ner'

        templates_dict = get_template()

        if model_name.find('BERT') > -1 or model_name.find('ALBERT') > -1:
            dev_data_bieo, test_data_bieo = data_concat(base_path)
        else:
            dev_data_bieo = read_predicted_corpus(dev_file_path)
            test_data_bieo = read_predicted_corpus(test_file_path)

        print('\ndev data baseline result of bieo:')
        dev_data_beo_eval = data2eval_data(dev_data_bieo)
        baseline_result = myevaluate(dev_data_beo_eval, base_path + 'dev_data_beo_eval', base_path + 'dev_beo_result_eval')
        f1_baseline = get_f1(baseline_result)

        print('\ntest data baseline result of bieo')
        test_data_beo_eval = data2eval_data(test_data_bieo)
        myevaluate(test_data_beo_eval, base_path + 'test_data_beo_eval', base_path + 'test_beo_result_eval')

        dev = data2word_pos_tag_tuple(dev_data_bieo)
        test = data2word_pos_tag_tuple(test_data_bieo)

        print('\n match dev data...')
        for chunk_class, templates in templates_dict.items():
            for init_template in templates:
                templates_extended = template_extend(init_template.elements, init_template.max_end_index)
                for template in templates_extended:
                    dev_cp = copy.deepcopy(dev)
                    print('\n template:',
                          [(element.index, element.name, element.value, element.relation) for element in template])
                    for one_sent_data in dev_cp:
                        match(one_sent_data, template, chunk_class)

                    print('\nthe performance of the template in dev data:')
                    dev_matched_eval_bieo = word_pos_tag_tuple2eval_data(dev_cp)
                    result_template = myevaluate(dev_matched_eval_bieo, base_path + 'deved_data_beo', base_path + 'deved_result_beo')
                    f1_template = get_f1(result_template)
                    if f1_template - f1_baseline >= threshold:
                        if template_selected.get(chunk_class) != None:
                            template_selected.get(chunk_class).append(template)
                        else:
                            template_temp = []
                            template_temp.append(template)
                            template_selected.update({chunk_class: template_temp})

        for chunk_class, templates in template_selected.items():
            for template in templates:
                print('\n template:',
                      [(element.index, element.name, element.value, element.relation) for element in template])
                for one_sent_data in dev:
                    match(one_sent_data, template, chunk_class)

        print('\nthe performance of sequence rule matched dev data bieo')
        dev_matched_eval_bieo = word_pos_tag_tuple2eval_data(dev)
        myevaluate(dev_matched_eval_bieo, base_path + 'deved_data_beo', base_path + 'deved_result_beo')

        print('\n match test data...')
        for chunk_class, templates in template_selected.items():
            for template in templates:
                print('\n template:',
                      [(element.index, element.name, element.value, element.relation) for element in template])
                for one_sent_data in test:
                    match(one_sent_data, template, chunk_class)

        print('\nthe performance of sequence rule matched test data bieo')
        test_matched_eval_bieo = word_pos_tag_tuple2eval_data(test)
        myevaluate(test_matched_eval_bieo, base_path + 'tested_data_beo', base_path + 'tested_result_beo')



def template_extend(template, max_end_index):
    templates_extended = []
    templates_extended.append(template)
    if template[-1].index >= max_end_index:
        return templates_extended
    extend_ele_idx = -1
    for element in template:
        extend_ele_idx += 1
        if element.extend == True:
            break

    for offset in range(1, max_end_index - template[-1].index + 1):
        template_new = copy.deepcopy(template)
        for element in template_new[extend_ele_idx:]:
            element.index += offset
        templates_extended.append(template_new)

    return templates_extended


def match(one_sent_data, template, chunk_class):
    token_matched = []
    match_count = 0
    idx_center_token = 0
    idx_ele = 0
    while idx_center_token < len(one_sent_data) and idx_ele < len(template):
        element = template[idx_ele]
        idx_token = idx_center_token + element.index
        if idx_token < 0 or idx_token >= len(one_sent_data):
            idx_center_token += 1
            continue
        else:
            one_token_data = one_sent_data[idx_token]
        if (element.relation == '=' and one_token_data[0][element.name] in element.value) or (
                element.relation == '!=' and one_token_data[0][element.name] not in element.value):
            match_count += 1
            idx_ele += 1
            token_matched.append(one_token_data)
            if match_count == len(template):
                # print(token_matched)
                match_count = 0
                token_matched.clear()
                template = [ele for ele in template if ele.index >= 0]
                sorted(template, key=lambda ele: ele.index, reverse=False)
                print([(one_token[0]['word'], one_token[0]['iob'], one_token[0]['tag_p']) for one_token in
                       one_sent_data[idx_token - template[-1].index - 1: idx_token + 1]])
                for idx_matched_token in range(idx_token - template[-1].index, idx_token + 1):
                    if idx_matched_token == idx_token - template[-1].index:
                        one_sent_data[idx_matched_token][1] = 'B-' + chunk_class
                    elif idx_matched_token == idx_token:
                        one_sent_data[idx_matched_token][1] = 'E-' + chunk_class
                    else:
                        one_sent_data[idx_matched_token][1] = 'I-' + chunk_class
        else:
            idx_center_token += 1
            idx_ele = 0
            match_count = 0
            token_matched.clear()

def write_data(data, label_path):
    """

        :param label_predict:
        :param label_path:
        :param metric_path:
        :return:
        """
    with open(label_path, "w") as fw:
        line = []
        for sent_result in data:
            for char, tag, tag_, pos in zip(sent_result[0], sent_result[1], sent_result[2], sent_result[3]):
                tag = '0' if tag == 'O' else tag
                # char = char.encode("utf-8")
                line.append("{} {}\n".format(char, tag))
            line.append("\n")
        fw.writelines(line)


def data_format(base_path):
    train_data = read_corpus(train_file_path)
    dev_data_bieo = read_predicted_corpus(dev_file_path)
    test_data_bieo = read_predicted_corpus(test_file_path)


    write_data(train_data, base_path+'/train.txt')
    write_data(dev_data_bieo, base_path + '/dev.txt')
    write_data(test_data_bieo, base_path+'/test.txt')

def data_concat(base_path):
    dev_data = []
    test_data = []

    dev_data_ori = read_corpus(base_path+'/dev_data')
    test_data_ori = read_corpus(base_path+'/test_data')

    dev_data_predicted = read_corpus_3(base_path+'/label_dev')
    test_data_predicted = read_corpus_3(base_path + '/label_test')

    for sent_, sent_predicted_ in zip(dev_data_ori, dev_data_predicted):
        dev_data.append([sent_predicted_[0], sent_predicted_[1], sent_predicted_[2], sent_[2], sent_[3]])

    for sent_, sent_predicted_ in zip(test_data_ori, test_data_predicted):
        test_data.append([sent_predicted_[0], sent_predicted_[1], sent_predicted_[2], sent_[2], sent_[3]])

    return dev_data, test_data


if __name__ == '__main__':
    train()
    # data_format()
