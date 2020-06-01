import os
import data

def myevaluate(model_predict, label_path, metric_path):
    all_result = []
    for _ in conlleval(model_predict, label_path, metric_path):
        print(_)
        all_result.append(_)
    return all_result

def write_model_predict_pos_ner(model_predict_pos_ner, label_path):
    """

        :param label_predict:
        :param label_path:
        :param metric_path:
        :return:
        """

    label_path = label_path + '_pos_ner'

    with open(label_path, "w") as fw:
        line = []
        for sent_result in model_predict_pos_ner:
            for char, tag, tag_, pos, ner in sent_result:
                tag = '0' if tag == 'O' else tag
                # char = char.encode("utf-8")
                line.append("{} {} {} {} {}\n".format(char, tag, tag_, pos, ner))
            line.append("\n")
        fw.writelines(line)


def conlleval(label_predict, label_path, metric_path):
    """

    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    eval_perl = "/home/robby/PycharmProjects/HybridChunkingModel/conlleval_rev.pl"
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                # char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics

def get_f1(all_result):
    for result in all_result:
        if 'accuracy' in result and 'FB1:' in result:
            return float(result[len(result)-5:])

    return 0