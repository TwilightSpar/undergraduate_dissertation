# -!- coding: utf-8 -!-
from transformers import BertForSequenceClassification
import torch
import pickle
from torch.utils.data import DataLoader
from transformers import AdamW

from success_bert import BertEncodingClassifier


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


def cal_F1(result, label):
    all_number = len(result)
    # print all_number
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(all_number):
        if result[i] == 1:
            if label[i][1] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if label[i][1] == 1:
                FN += 1
            else:
                TN += 1
    # print TP+FP+TN+FN
    accuracy = float(TP+TN) / float(all_number)
    P = float(TP) / float(TP + FP)
    R = float(TP) / float(TP + FN)
    F1 = 2/(1/P + 1/R)
    # print accracy, precision, TPR, TNR, FNR, FPR
    return F1, accuracy


train_dataset = None
with open('data/train_dataset', 'rb') as f:
    train_dataset = pickle.load(f)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BertEncodingClassifier.from_pretrained('bert-base-chinese')
# print(model)
model.to(device)
model.train()

# freeze the parameters and weight decay
lr_para = ['bert.encoder.layer.10', 'bert.encoder.layer.11', 'bert.pooler.dense', 'classifier']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if "bert.encoder.layer.11" in n], 'lr': 5e-5},
    {'params': [p for n, p in model.named_parameters() if "bert.encoder.layer.10" in n], 'lr': 1e-5}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-2)
for name, param in model.named_parameters():
    if not any(nd in name for nd in lr_para):
        param.requires_grad = False
########

result_list = []
label_list = []
for epoch in range(3):
    i = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        one_hot = batch['one_hot'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, one_hot=one_hot)
        result_list.extend(torch.argmax(outputs[1], 1).tolist())
        label_list.extend(labels.tolist())

        loss = outputs[0]
        loss.backward()
        optimizer.step()
        print("finished batch %d, loss: %f" % (i, loss))
        i = i + 1

        F1, acc = cal_F1(result_list, label_list)
        print("F1: %f ; acc: %f" % (F1, acc))

model.eval()
F1, acc = cal_F1(result_list, label_list)
print("F1: %f ; acc: %f" % (F1, acc))
print("finish fine tuning")
model.save_pretrained("./model/first_try")
