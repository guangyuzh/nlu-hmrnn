from nltk.corpus import treebank
import re, string
import os, argparse


PUNC_TABLE = str.maketrans({key: None for key in string.punctuation})

def flatten_tree(t, threshold):
    pos_list = t.pos()
    if len(pos_list) < threshold:
        return
    sentence = ""
    phrase = ""
    last_pos = ""
    for word, pos in pos_list:
        if word in string.punctuation:
            continue
        word = re.sub(r'\d', 'x', word)
        if last_pos != pos:
            if last_pos:
                sentence += phrase + '1'
            phrase = word
            last_pos = pos
        else:
            phrase += '0' + word
    try:
        if last_pos == pos_list[-2][1]:
            sentence += phrase
    except:
        print(pos_list)
    sentence = sentence.translate(PUNC_TABLE)
    binarify = re.compile(r'[a-z]', re.IGNORECASE)
    return binarify.sub('0', sentence)


def gen_corpus(path, threshold):
    # src: http://www.nltk.org/_modules/nltk/tree.html
    # corpora from wsj_0001.mrg to wsj_0199.mrg
    # e.g.: t = treebank.parsed_sents('wsj_0001.mrg')[0]
    # t.draw()
    boundaries = []
    sentences = []
    for file in treebank.parsed_sents(treebank.fileids()):
        for t in file:
            flat = flatten_tree(t, threshold)
            if flat:
                boundaries.append(flat)
                sentence = ' '.join(t.leaves()).translate(PUNC_TABLE)
                sentence = re.sub(r' +', ' ', sentence)
                # replace digit(s) as 'x'(s)
                sentences.append(re.sub(r'\d', 'x', sentence))
    with open(path + "/boundaries.txt", 'w') as f:
        f.write('\n'.join(boundaries))
    with open(path + "/sentences.txt", 'w') as f:
        f.write('\n'.join(sentences))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', action='store', dest='path', default='corpora',
                        help='output path for storing generated corpus')
    parser.add_argument('--threshold', action='store', dest='threshold', default=5,
                        help='minimum number of tokens (including word and punctuation) in a sentence')
    options = parser.parse_args()

    if not os.path.exists(options.path):
        os.makedirs(options.path)

    gen_corpus(**vars(options))