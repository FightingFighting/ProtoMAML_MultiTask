from torch import nn
from transformers import RobertaModel

class EmoClassifier_MulTask(nn.Module):

  def __init__(self, pretrained_model_name, n_classes, emotions):
    super(EmoClassifier_MulTask, self).__init__()
    self.encoder = RobertaModel.from_pretrained(pretrained_model_name)
    self.drop = nn.Dropout(p=0.3)
    if len(n_classes)==1:
      self.fc_layer_task0 = nn.Linear(self.encoder.config.hidden_size, n_classes[0])
    if len(n_classes)==2:
      self.fc_layer_task0 = nn.Linear(self.encoder.config.hidden_size, n_classes[0])
      self.fc_layer_task1 = nn.Linear(self.encoder.config.hidden_size, n_classes[1])
    if len(n_classes)==3:
      self.fc_layer_task0 = nn.Linear(self.encoder.config.hidden_size, n_classes[0])
      self.fc_layer_task1 = nn.Linear(self.encoder.config.hidden_size, n_classes[1])
      self.fc_layer_task2 = nn.Linear(self.encoder.config.hidden_size, n_classes[2])
    if len(n_classes)==4:
      self.fc_layer_task0 = nn.Linear(self.encoder.config.hidden_size, n_classes[0])
      self.fc_layer_task1 = nn.Linear(self.encoder.config.hidden_size, n_classes[1])
      self.fc_layer_task2 = nn.Linear(self.encoder.config.hidden_size, n_classes[2])
      self.fc_layer_task3 = nn.Linear(self.encoder.config.hidden_size, n_classes[3])

    self.fc_layer_allTask={}
    for ind, e in enumerate(emotions):
      self.fc_layer_allTask[e] = eval(f'self.fc_layer_task{ind}')

  def forward(self, input_ids, attention_mask, emotion):
    output_tuple = self.encoder(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    pooled_output = output_tuple.pooler_output
    output = self.drop(pooled_output)
    fc = self.fc_layer_allTask[emotion]
    output = fc(output)

    return output
