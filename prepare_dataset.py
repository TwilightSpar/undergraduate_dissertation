# -!- coding: utf-8 -!-
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import pickle
# the max len of the intro is 1830 words


def read_intro():
    texts = []
    one_hot = torch.empty(size=[0, 21])
    labels = []
    df = None
    with open('data/learning_data_all', 'rb') as f:
        df = pickle.load(f)

    for index, row in df.iterrows():
        if row["success"] == 1:
            labels.append([0, 1])
        elif row["success"] == 0:
            labels.append([1, 0])
        else:
            continue

        texts.append(row["intro"])
        exit_time = row["exit_time"]
        field = row["field"]
        loc = row["loc"]
        one_row_encode = torch.cat((exit_time, field, loc), 1)
        one_hot = torch.cat((one_hot, one_row_encode), 0)
    one_hot = one_hot.tolist()
    return texts, one_hot, labels


train_text, train_one_hot, train_label = read_intro()
train_data = list(zip(train_text, train_one_hot))
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_label, test_size=.2)

train_text, train_one_hot = zip(*train_data)
test_text, test_one_hot = zip(*test_data)
print(len(test_labels))  # 642
train_text = list(train_text)
train_one_hot = list(train_one_hot)
test_text = list(test_text)
test_one_hot = list(test_one_hot)

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
tokenizer.model_max_length = 512
# train_encodings = tokenizer("train texts fast", truncation=True, padding=True)
# print(tokenizer.get_vocab())
# print(tokenizer.pad_token_id)
# train_encodings = tokenizer(train_texts, max_length=512) # , truncation=True, padding=True
# test_encodings = tokenizer(test_texts, max_length=512) # , padding=True
train_encodings = tokenizer(train_text, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_text, truncation=True, padding=True, max_length=512)

print("finish text data")


class IntroDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, one_hot, labels):
        self.encodings = encodings
        self.labels = labels
        self.one_hot = one_hot

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.int64)
        item['one_hot'] = torch.tensor(self.one_hot[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IntroDataset(train_encodings, train_one_hot, train_labels)
test_dataset = IntroDataset(test_encodings, test_one_hot, test_labels)


with open('data/train_dataset', 'wb') as file:
    pickle.dump(train_dataset, file)

with open('data/test_dataset', 'wb') as file:
    pickle.dump(test_dataset, file)
