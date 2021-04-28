# -!- coding: utf-8 -!-
import math

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import pickle
import torch.nn.functional as F


class IntroDataset(torch.utils.data.Dataset):
    def __init__(self, one_hot, labels):
        self.labels = labels
        self.one_hot = one_hot

    def __getitem__(self, idx):
        item = {'labels': torch.tensor(self.labels[idx], dtype=torch.int64), 'one_hot': torch.tensor(self.one_hot[idx])}
        return item

    def __len__(self):
        return len(self.labels)


def FocalLoss(inputs, targets):
    alpha = 0.47
    at = torch.tensor([alpha, 1-alpha]).gather(0, targets.data.view(-1))
    at = at.reshape([-1, 2])
    gamma = 5

    BCE_loss = F.binary_cross_entropy(inputs, targets.float(), reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = at * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()


def read_intro():
    one_hot = torch.empty(size=[0, 21])
    labels = []
    df = None
    with open('data/learning_data_all', 'rb') as f:
        df = pickle.load(f)

    for index, row in df.iterrows():
        # discard label 2: can not decide the class
        if row["success"] == 1:
            labels.append([0, 1])
        elif row["success"] == 0:
            labels.append([1, 0])
        else:
            continue
        exit_time = row["exit_time"]
        field = row["field"]
        loc = row["loc"]
        one_row_encode = torch.cat((exit_time, field, loc), 1)
        one_hot = torch.cat((one_hot, one_row_encode), 0)
    one_hot = one_hot.tolist()
    return one_hot, labels


train_one_hot, train_label = read_intro()
train_one_hot, test_one_hot, train_labels, test_labels = train_test_split(train_one_hot, train_label, test_size=.2)

print(len(test_labels))  # 927

train_dataset = IntroDataset(train_one_hot, train_labels)
test_dataset = IntroDataset(test_one_hot, test_labels)


with open('data/train_onehot_dataset', 'wb') as file:
    pickle.dump(train_dataset, file)

with open('data/test_onehot_dataset', 'wb') as file:
    pickle.dump(test_dataset, file)


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
    # print accuracy, precision, TPR, TNR, FNR, FPR
    return F1, accuracy


def f1_loss(inputs, targets):
    all_number = len(inputs)
    # print all_number
    tp = 0.
    fp = 0.
    fn = 0.
    tn = 0.
    for i in range(all_number):
        pre_class = torch.argmax(inputs[i])
        if pre_class == 1:
            tp += inputs[i][1] * targets[i][1]
            fp += inputs[i][1] * targets[i][0]
        else:
            fn += inputs[i][0] * targets[i][1]
            tn += inputs[i][0] * targets[i][0]

    p = tp / (tp + fp + 1e-5)
    r = tp / (tp + fn + 1e-5)

    f1 = 2 * p * r / (p + r + 1e-5)
    return 1 - f1


train_dataset = None
with open('data/train_onehot_dataset', 'rb') as f:
    train_dataset = pickle.load(f)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(21, 21)
        self.fc2 = torch.nn.Linear(21, 10)
        self.fc3 = torch.nn.Linear(10, 2)

    def forward(self, one_hot, labels):
        din = torch.tensor(one_hot, dtype=torch.float).view(-1, 21)
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        logits = F.softmax(self.fc3(dout))

        # loss_fct = torch.nn.CrossEntropyLoss()
        loss = FocalLoss(logits.view(-1, 2), labels.view(-1, 2))
        return loss, logits


model = MLP()
model.to(device)
model.train()
# loss func and optim
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

result_list = []
label_list = []
for epoch in range(3):
    i = 0
    for batch in train_loader:
        optimizer.zero_grad()
        labels = batch['labels'].to(device)
        one_hot = batch['one_hot'].to(device)

        loss, logits = model(one_hot, labels)
        result_list.extend(torch.argmax(logits, 1).tolist())
        label_list.extend(labels.tolist())

        print(labels.tolist())
        # print(logits.tolist())

        loss.backward()
        optimizer.step()
        print("finished batch %d, loss: %f" % (i, loss))
        i = i + 1


F1, acc = cal_F1(result_list, label_list)
print("F1: %f ; acc: %f" % (F1, acc))
print("finish training")

model.eval()
result_list = []
label_list = []
i = 0

test_dataset = None
with open('data/test_onehot_dataset', 'rb') as f:
    test_dataset = pickle.load(f)
test_loader = DataLoader(test_dataset, batch_size=16)


for batch in test_loader:
    optimizer.zero_grad()
    labels = batch['labels'].to(device)
    one_hot = batch['one_hot'].to(device)

    loss, logits = model(one_hot, labels)
    result_list.extend(torch.argmax(logits, 1).tolist())
    label_list.extend(labels.tolist())
    print("finished batch %d, loss: %f" % (i, loss))
    i = i+1

F1, acc = cal_F1(result_list, label_list)
print("F1: %f ; acc: %f" % (F1, acc))
