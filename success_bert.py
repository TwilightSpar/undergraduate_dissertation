import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F


def FocalLoss(inputs, targets):
    alpha = 0.31
    at = torch.tensor([alpha, 1-alpha]).gather(0, targets.data.view(-1))
    at = at.reshape([-1, 2])
    gamma = 2

    BCE_loss = F.binary_cross_entropy(inputs, targets.float(), reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = at * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()


class BertEncodingClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(in_features=768 + 21, out_features=2)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            one_hot=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        new_output = torch.cat((pooled_output, one_hot), dim=1)
        logits = F.softmax(self.classifier(new_output))
        loss = FocalLoss(logits.view(-1, 2), labels.view(-1, 2))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
