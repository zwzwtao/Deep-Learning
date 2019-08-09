import codecs
import collections
from operator import itemgetter

# set MODE to "PTB" or "TRANSLATE_EN" or "TRANSLATE_ZH"
MODE = "PTB"

if MODE == "PTB":
    RAW_DATA = "PTB_data/ptb.train.txt"
    VOCAB_OUTPUT = "ptb.vocab"
elif MODE == "TRANSLATE_ZH":
    RAW_DATA = "TED_data/train.txt.zh"
    VOCAB_OUTPUT = "zh.vocab"
    VOCAB_SIZE = 4000
elif MODE == "TRANSLATE_EN":
    RAW_DATA = "TED_Data/train.txt.en"
    VOCAB_OUTPUT = "en.vocab"
    VOCAB_SIZE = 10000

# count the frequency of each word
counter = collections.Counter()
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1
# print(counter.items())

# sort words on frequency
sorted_word_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
# sorted_word_cnt: [('the', 50770), ('<unk>', 45020), ('N', 32481),......]
sorted_words = [x[0] for x in sorted_word_cnt]

if MODE == "PTB":
    # add <eos> at text line break
    sorted_words = ["<eos>"] + sorted_words
elif MODE in ["TRANSLATE_EN", "TRANSLATE_ZN"]:
    # also need to add <unk> and <sos> to "TRANSLATE_EN", "TRANSLATE_ZN", here we ignore "PTB"
    # then delete words with low frequency
    sorted_words = ["unk", "<sos>", "<eos>"] + sorted_words
    if len(sorted_words) > VOCAB_SIZE:
        sorted_words = sorted_words[: VOCAB_SIZE]

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')
