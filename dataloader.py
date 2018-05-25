import json
import re
import collections
import pdb

class DataLoader():
    def __init__(self, file_path):
        self._pad_word = "<pad>"
        self._pad_label = "label:pad"
        self._file_path = file_path

        word_to_id, vocab_size, label_to_id, label_size, id_to_word, id_to_label, sentence_max_len = self._build_vocab()

        word_to_id[self._pad_word] = 0
        label_to_id[self._pad_label] = 0

        self._word_to_id = word_to_id
        # + 1 for padding
        self._vocab_size = vocab_size + 1
        self._label_to_id = label_to_id
        self._label_size = label_size + 1
        self._id_to_word = id_to_word
        self._id_to_label = id_to_label
        self._sentence_max_len = sentence_max_len

    def _normalize_sentence(self, sentence):
        return re.sub(' +', ' ', sentence.lower().strip())

    def _read_example_by_word(self, json_string):
        words, labels = self._read_example_by_sentence(json_string)
        return list(zip(words, labels))

    def _read_example_by_sentence(self, json_string):
        data = json.loads(self._normalize_sentence(json_string))
        raw_words = data["input"]
        raw_labels = data["output"]
        assert len(raw_words) == len(raw_labels)

        parse_words = list()
        parse_labels = list()

        for index, word in enumerate(raw_words):
            parse = word.split()
            parse_words.extend(parse)
            for _ in parse:
                parse_labels.append(raw_labels[index])

        assert len(parse_words) == len(parse_labels)
        # pdb.set_trace()
        return parse_words, parse_labels, len(parse_words)

    def _build_vocab(self):
        with open(self.file_path, 'r') as f:
            sentence_max_len = 0
            lines = f.readlines()
            all_word = []
            all_label = []
            for line in lines:
                words, labels, leng = self._read_example_by_sentence(line)
                all_word.extend(words)
                all_label.extend(labels)
                sentence_max_len = sentence_max_len if leng < sentence_max_len else leng
                # pdb.set_trace()

            word_counter = collections.Counter(all_word)
            label_counter = collections.Counter(all_label)

            word_count_pairs = sorted(word_counter.items(), key=lambda x: (-x[1], x[0]))
            label_count_pairs = sorted(label_counter.items(), key=lambda x: (-x[1], x[0]))
            words, _ = list(zip(*word_count_pairs))
            labels, _ = list(zip(*label_count_pairs))

            word_to_id = dict(zip(words, range(1, len(words) + 1)))
            label_to_id = dict(zip(labels, range(1, len(words) + 1)))
            id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
            id_to_label = dict(zip(label_to_id.values(), label_to_id.keys()))
            # import pdb; pdb.set_trace()
            return word_to_id, len(words), label_to_id, len(labels), id_to_word, id_to_label, sentence_max_len

    def _build_input(self):
        input = []
        return input

    def _example_to_id(self, example):
        inputs, outputs, leng = self._read_example_by_sentence(example)
        # convert to id
        inputs_to_id = [self.word_to_id[word] for word in inputs]
        outputs_to_id = [self.label_to_id[label] for label in outputs]
        return inputs_to_id, outputs_to_id, leng

    # return list of data
    def load_data(self, number_data, train_data, test_data):
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            dataset = list()
            for line in lines:
                input_to_id, output_to_id, leng = self._example_to_id(line)
                # add padding
                for _ in range(leng + 1, self.sentence_max_len + 1):
                    input_to_id.append(self.word_to_id[self._pad_word])
                    output_to_id.append(self.label_to_id[self._pad_label])
                dataset.append([input_to_id, output_to_id])

            return dataset[:int(number_data * train_data)], dataset[int(number_data * train_data): number_data]

    @property
    def word_to_id(self):
        return self._word_to_id

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def label_to_id(self):
        return self._label_to_id

    @property
    def label_size(self):
        return self._label_size

    @property
    def id_to_word(self):
        return self._id_to_word

    @property
    def id_to_label(self):
        return self._id_to_label

    @property
    def sentence_max_len(self):
        return self._sentence_max_len

    @property
    def file_path(self):
        return self._file_path
