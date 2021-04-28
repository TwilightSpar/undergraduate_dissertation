from transformers import AdamW

from success_bert import BertEncodingClassifier

model = BertEncodingClassifier.from_pretrained('bert-base-chinese')
# print(model)

lr_para = ['bert.encoder.layer.10', 'bert.encoder.layer.11', 'bert.pooler.dense', 'classifier']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if "bert.encoder.layer.11" in n], 'lr': 1e-3},
    {'params': [p for n, p in model.named_parameters() if "bert.encoder.layer.10" in n], 'lr': 1e-4}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-2)
for name, param in model.named_parameters():
    if not any(nd in name for nd in lr_para):
        param.requires_grad = False

for name, param in model.named_parameters():
    print("%-60s " % name, param.requires_grad)

