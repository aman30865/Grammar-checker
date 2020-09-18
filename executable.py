# Loading the Model

from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch

# https://drive.google.com/drive/folders/1gBtM3ifYsrZ70ZmNW1hj-pb-dxvd2n3b?usp=sharing
# download the saved model using the above link.

save_dir = './saved_model/'

tokenizer = BertTokenizer.from_pretrained(save_dir)
model_loaded = BertForSequenceClassification.from_pretrained(sav_dir)

# Prediction

again = 1
while(again):
  sent = input()
  encoded_dict = tokenizer.encode_plus(
                          sent,
                          add_special_tokens = True,
                          max_length = 64,
                          truncation=True,
                          padding='max_length',
                          return_attention_mask = True,
                          return_tensors = 'pt',
                    )
  input_id = encoded_dict['input_ids']
  attention_mask = encoded_dict['attention_mask']
  input_id = torch.LongTensor(input_id)
  attention_mask = torch.LongTensor(attention_mask)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_loaded = model_loaded.to(device)
  input_id = input_id.to(device)
  attention_mask = attention_mask.to(device)
  with torch.no_grad():
    outputs = model_loaded(input_id, token_type_ids=None, attention_mask=attention_mask)
  logits = outputs[0]
  index = logits.argmax()
  if index == 1:
    print("The given sentence is Gramatically correct")
  else:
    print("The given sentence is Gramatically in-correct")
  again=int(input("Enter 1 to Continue or 0 to Exit:"))
