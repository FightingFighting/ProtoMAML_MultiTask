from torch import nn
from transformers import RobertaModel

class EmoClassifier(nn.Module):

  def __init__(self, pretrained_model_name, n_classes):
    super(EmoClassifier, self).__init__()
    self.encoder = RobertaModel.from_pretrained(pretrained_model_name)
    self.drop = nn.Dropout(p=0.3)
    self.fc_layer = nn.Linear(self.encoder.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    output_tuple = self.encoder(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    pooled_encodings = output_tuple.pooler_output
    #pooled_encodings = self.drop(pooled_encodings)
    output = self.fc_layer(pooled_encodings)
    return output, pooled_encodings