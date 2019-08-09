import codecs
import sys

MODE = "PTB_TRAIN"

if MODE == "PTB_TRAIN":
    RAW_DATA = "PTB_data/ptb.train.txt"
    VOCAB = "ptb.vocab"
    OUTPUT_DATA = "ptb.train"
elif MODE == "PTB_VALID":
    RAW_DATA = "PTB_data/ptb.valid.txt"
    VOCAB = "ptb.vocab"
    OUTPUT_DATA = "ptb.valid"
elif MODE == "PTB_TEST":
    RAW_DATA = "PTB_data/ptb.test.txt"
    VOCAB = "ptb.vocab"
    OUTPUT_DATA = "ptb.test"
elif MODE == "TRANSLATE_ZH":
    RAW_DATA = "TED_data/train.txt.zh"
    VOCAB = "zh.vocab"
    OUTPUT_DATA = "train.zh"
elif MODE == "TRANSLATE_EN":
    RAW_DATA = "TED_data/train.txt.en"
    VOCAB = "en.vocab"
    OUTPUT_DATA = "train.en"

with codecs.open(VOCAB, 'r', "utf-8") as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
# create a map from word to its index
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}


# replace words not in the vocab with "unk"
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']


fin = codecs.open(RAW_DATA, "r", "utf-8")
fout = codecs.open(OUTPUT_DATA, "w", "utf-8")
for line in fin:
    # read words and add <eos> at the end
    words = line.strip().split() + ["<eos>"]
    output_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(output_line)
fin.close()
fout.close()
