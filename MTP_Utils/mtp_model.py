import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForSequenceClassification
from transformers import RobertaModel, RobertaForSequenceClassification

fchidden = 768
hiddendim_lstm = 128
embeddim = 768
numlayers = 12
roberta_checkname =  'cardiffnlp/twitter-roberta-base-2021-124m-sentiment'


class Bert_Base(nn.Module):
    def __init__(self, numclasses):
        super(Bert_Base, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.dropout = nn.Dropout(0.1)

        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', # noqa
                                                                   output_hidden_states=False, # noqa
                                                                   output_attentions=False, # noqa
                                                                   num_labels=self.numclasses) # noqa
        print("BERT Model Loaded")

    def forward(self, inp_ids, att_mask, token_ids, labels):

        out = self.bert(input_ids=inp_ids, 
                        attention_mask=att_mask,
                        token_type_ids=token_ids, 
                        labels=labels)
                        
        loss = out.loss
        logits = out.logits

        return loss, logits


class Roberta_Base(nn.Module):
    def __init__(self, numclasses):
        super(Roberta_Base, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.dropout = nn.Dropout(0.1)

        self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-base', # noqa
                                                                         output_hidden_states=False, # noqa
                                                                         output_attentions=False, # noqa
                                                                         num_labels=self.numclasses) # noqa
        print("😉---RoBERTa Model Loaded---😉")

    def forward(self, inp_ids, att_mask, token_ids, labels):

        out = self.roberta(input_ids=inp_ids, 
                           attention_mask=att_mask,
                           token_type_ids=token_ids, 
                           labels=labels)
                        
        loss = out.loss
        logits = out.logits

        return loss, logits


class Bert_LSTM(nn.Module):
    def __init__(self, numclasses):
        super(Bert_LSTM, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.numlayers = numlayers
        self.hiddendim_lstm = hiddendim_lstm
        self.dropout = nn.Dropout(0.1)

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_hidden_states=True,
                                              output_attentions=False)
        print("BERT Model Loaded")
        self.lstm = nn.LSTM(self.embeddim, self.hiddendim_lstm, batch_first=True) # noqa
        self.fc = nn.Linear(self.hiddendim_lstm, self.numclasses)

    def forward(self, inp_ids, att_mask, token_ids):
        last_hidden_state, pooler_output, \
                hidden_states = self.bert(input_ids=inp_ids,
                                          attention_mask=att_mask,
                                          token_type_ids=token_ids)

        hidden_states = torch.stack([hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(0, self.numlayers)], dim=-1) # noqa
        hidden_states = hidden_states.view(-1, self.numlayers, self.embeddim)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class Bert_Attention(nn.Module):
    def __init__(self, numclasses, device):
        super(Bert_Attention, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.numlayers = numlayers
        self.fchidden = fchidden
        self.dropout = nn.Dropout(0.1)

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_hidden_states=True,
                                              output_attentions=False)
        print("BERT Model Loaded")

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.embeddim))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.embeddim, self.fchidden)) # noqa
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(device)

        self.fc = nn.Linear(self.fchidden, self.numclasses)

    def forward(self, inp_ids, att_mask, token_ids):
        out_bert = self.bert(input_ids=inp_ids.long(), 
                             attention_mask=att_mask.long(), 
                             token_type_ids=token_ids.long())

        last_hidden_state = out_bert.last_hidden_state
        pooler_output = out_bert.pooler_output
        hidden_states = out_bert.hidden_states

        hidden_states = torch.stack([hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(0, self.numlayers)], dim=-1) # noqa
        hidden_states = hidden_states.view(-1, self.numlayers, self.embeddim)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v


class Roberta_Attention(nn.Module):
    def __init__(self, numclasses, device):
        super(Roberta_Attention, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.numlayers = numlayers
        self.fchidden = fchidden
        self.dropout = nn.Dropout(0.3)

        self.roberta = RobertaModel.from_pretrained(roberta_checkname,
                                                    output_hidden_states=True,
                                                    output_attentions=False)
        print("😉---RoBERTa Model Loaded---😉")

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.embeddim))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.embeddim, self.fchidden)) # noqa
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(device)

        self.fc_0 = nn.Linear(self.fchidden, self.fchidden)
        self.act1 = nn.ReLU()
        self.fc = nn.Linear(self.embeddim, self.numclasses)

    def forward(self, inp_ids, att_mask, token_ids):
        out_roberta = self.roberta(input_ids=inp_ids.long(), 
                                   attention_mask=att_mask.long(), 
                                   token_type_ids=token_ids.long())

        last_hidden_state = out_roberta.last_hidden_state
        pooler_output = out_roberta.pooler_output
        hidden_states = out_roberta.hidden_states

        hidden_states = torch.stack([hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(0, self.numlayers)], dim=-1) # noqa
        hidden_states = hidden_states.view(-1, self.numlayers, self.embeddim)
#         out = self.attention(hidden_states)
#         out = self.dropout(out)
#         out = self.fc_0(out)
#         out = self.act1(out)
#         out = self.dropout(out)
        out = self.dropout(pooler_output)
        out = self.fc(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v

class Roberta_Denoiser(nn.Module):
    def __init__(self, numclasses, device, rode_model_path):
        super(Roberta_Denoiser, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.numlayers = numlayers
        self.fchidden = fchidden
        self.dropout = nn.Dropout(0.1)
        
        pretrained_model = torch.load(rode_model_path)
        rode_model = RobertaModel.from_pretrained(roberta_checkname,
                                                  output_hidden_states=True,
                                                  output_attentions=False)
        pretrained_model = pretrained_model.to(device)
        rode_model = rode_model.to(device)
        
        pretrained_dict = pretrained_model.state_dict()
        rode_dict = rode_model.state_dict()
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in rode_dict}
        # 2. overwrite entries in the existing state dict
        rode_dict.update(pretrained_dict)
        # 3. load the new state dict
        rode_model.load_state_dict(rode_dict)

        self.roberta = rode_model
        print("😉---RoBERTa Model Loaded---😉")


    def forward(self, inp_ids, att_mask, token_ids):

        out_roberta = self.roberta(input_ids=inp_ids.long(), 
                                   attention_mask=att_mask.long(), 
                                   token_type_ids=token_ids.long())

        last_hidden_state = out_roberta.last_hidden_state
        pooler_output = out_roberta.pooler_output
        hidden_states = out_roberta.hidden_states

        return pooler_output, hidden_states

class Roberta_Comp(nn.Module):
    def __init__(self, numclasses, device, rode_model_path):
        super(Roberta_Comp, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.numlayers = numlayers
        self.fchidden = fchidden
        self.dropout = nn.Dropout(0.1)

        pretrained_model = torch.load(rode_model_path)
        rode_model = RobertaModel.from_pretrained(roberta_checkname,
                                                  output_hidden_states=True,
                                                  output_attentions=False)
        pretrained_model = pretrained_model.to(device)
        rode_model = rode_model.to(device)
        
        pretrained_dict = pretrained_model.state_dict()
        rode_dict = rode_model.state_dict()
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in rode_dict}
        # 2. overwrite entries in the existing state dict
        rode_dict.update(pretrained_dict)
        # 3. load the new state dict
        rode_model.load_state_dict(rode_dict)

        self.roberta = rode_model
        print("😉---RoBERTa Model Loaded---😉")


    def forward(self, inp_ids, att_mask, token_ids):
      

        out_roberta = self.roberta(input_ids=inp_ids.long(), 
                                   attention_mask=att_mask.long(), 
                                   token_type_ids=token_ids.long())

        last_hidden_state = out_roberta.last_hidden_state
        pooler_output = out_roberta.pooler_output
        hidden_states = out_roberta.hidden_states

        out = pooler_output

        return out

# class AutoEncoder(nn.Module):
#     def __init__(self, device):
#         super(AutoEncoder, self).__init__()
        
#         self.encoder_layer1 = nn.Sequential(
# #             nn.Dropout(0.1),
#             nn.Linear(768, 256), nn.Mish(),
#             nn.Linear(256, 128), nn.Mish(),
#             nn.Linear(128, 64), nn.Mish(), 
#         )
#         self.encoder_layer2 = nn.Sequential(
# #             nn.Dropout(0.1),
#             nn.Linear(64, 32), nn.Mish(),
#             nn.Linear(32, 24), nn.Mish(),
#             nn.Linear(24, 12), nn.Mish(),
#         )
#         self.decoder_layer1 = nn.Sequential(
# #             nn.Dropout(0.1),
#             nn.Linear(12, 24), nn.Mish(),
#             nn.Linear(24, 32), nn.Mish(),
#             nn.Linear(32, 64), nn.Mish(),
#         )
#         self.decoder_layer2 = nn.Sequential(
# #             nn.Dropout(0.1),
#             nn.Linear(64, 128), nn.Mish(),
#             nn.Linear(128, 256), nn.Mish(),
#             nn.Linear(256, 768), nn.Mish(),
#         )
#         self.linear = nn.Sequential(
# #             nn.Dropout(0.3),
#             nn.Linear(768, 768), nn.Tanh()
#         )
#         self.norm = nn.BatchNorm1d(768)
#         self.norm_en1 = nn.BatchNorm1d(64)
#         self.norm_en2 = nn.BatchNorm1d(12)
#         self.norm_de1 = nn.BatchNorm1d(64)
#         self.norm_de2 = nn.BatchNorm1d(768)
#         self.dropout =  nn.Dropout(0.3)


#     def forward(self, x):
# #         x = self.dropout(x)
#         x = self.norm(x) # batch normalization
#         x_shortcut = x.clone() # 1st skip connection
#         out_encoder1 = self.encoder_layer1(x)
#         ec1_shortcut = out_encoder1.clone() # 2st skip connection
#         out_encoder2 = self.encoder_layer2(out_encoder1)
#         out_decoder1 = self.decoder_layer1(out_encoder2)
#         out_decoder2 = self.decoder_layer2(torch.add(out_decoder1, ec1_shortcut)) # 2st skip connection
# #         out_decoder2 = self.norm_de2(out_decoder2) # batch normalization
#         out_linear = self.linear(torch.add(out_decoder2, x_shortcut)) # 1st skip connection
#         return out_encoder2, out_linear


class AutoEncoder(nn.Module):
    def __init__(self, device):
        super(AutoEncoder, self).__init__()
        self.encoder_layer1 = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 24), nn.ReLU(),
            nn.Linear(24, 12), 
        )
        self.decoder_layer1 = nn.Sequential(
            nn.Linear(12, 24), nn.ReLU(),
            nn.Linear(24, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
        )
        self.decoder_layer2 = nn.Sequential(
#             nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 768), 
#             nn.ReLU()
        )
        self.linear = nn.Sequential(
#             nn.BatchNorm1d(768),
            nn.Linear(768, 768), 
            nn.Tanh()
        )
        self.norm = nn.BatchNorm1d(768)

    def forward(self, x):
        x = self.norm(x)
        x_shortcut = x.clone() # 1st skip connection
        out_encoder1 = self.encoder_layer1(x)
        ec1_shortcut = out_encoder1.clone() # 2nd skip connection
        out_encoder2 = self.encoder_layer2(out_encoder1)
        out_decoder1 = self.decoder_layer1(out_encoder2)
        out_decoder2 = self.decoder_layer2(torch.add(out_decoder1, ec1_shortcut)) # 2nd skip connection
        out_linear = self.linear(torch.add(out_decoder2, x_shortcut)) # 1st skip connection
        return out_encoder2, out_linear


class SentimentClassifier(nn.Module):

  def __init__(self, n_classes, rode_model_path, aede_model_path):
    super(SentimentClassifier, self).__init__()

    self.numclasses = n_classes
    self.embeddim = embeddim
    self.path_rode = rode_model_path
    self.path_aede = aede_model_path
    
    # load RoBERTa Denoiser (rode)
    # load AutoEncoder Denoiser (aede)
    self.rode_model = torch.load(self.path_rode)
    self.aede_model = torch.load(self.path_aede)

    self.drop = nn.Dropout(p=0.3)
    self.linear1 = nn.Linear(self.embeddim, 256)
    self.norm = nn.BatchNorm1d(256)
    self.act1 = nn.ReLU()
    self.linear2 = nn.Linear(256, self.numclasses)

  
  def forward(self, inp_ids, att_mask, token_ids):

    rode_output,_ = self.rode_model(inp_ids, 
                                    att_mask,
                                    token_ids)
    _,aede_output = self.aede_model(rode_output)

    output = self.drop(aede_output)
    output = self.linear1(output)
    # output = self.norm(output)
    output = self.act1(output)
    output = self.drop(output)
    output = self.linear2(output)


    return output


class SentimentCruiser(nn.Module):

  def __init__(self, n_classes, device, rode_model_path, aede_model_path):
    super(SentimentCruiser, self).__init__()

    self.numclasses = n_classes
    self.embeddim = embeddim
    self.numlayers = numlayers
    self.fchidden = fchidden
    self.dropout = nn.Dropout(0.2)
    self.at_norm = nn.BatchNorm1d(self.fchidden)
    self.ae_norm = nn.BatchNorm1d(self.embeddim)
    
    self.conv_params = {"stride":2,
                        "out_channels":96, 
                        "kernel_size_1":4, 
                        "kernel_size_2":6,
                        "input1_length":self.fchidden, # attention output size
                        "input2_length":self.embeddim, # AE output size
                       }
    self.pool_params = {"pool_size_11":int(1+(self.conv_params["input1_length"]\
                                              -(self.conv_params["kernel_size_1"]-1)-1)\
                                           /self.conv_params["stride"]),
                        "pool_size_12":int(1+(self.conv_params["input1_length"]\
                                              -(self.conv_params["kernel_size_2"]-1)-1)\
                                           /self.conv_params["stride"]),
                        "pool_size_21":int(1+(self.conv_params["input2_length"]\
                                              -(self.conv_params["kernel_size_1"]-1)-1)\
                                           /self.conv_params["stride"]),
                        "pool_size_22":int(1+(self.conv_params["input2_length"]\
                                              -(self.conv_params["kernel_size_2"]-1)-1)\
                                           /self.conv_params["stride"])
                       }
    
    
    self.dense_size = self.conv_params["out_channels"] * 8
    self.hidden_size = 256
#     self.norm = nn.BatchNorm1d(256)
    self.path_rode = rode_model_path
    self.path_aede = aede_model_path
    
    # load RoBERTa Denoiser (rode)
    # load AutoEncoder Denoiser (aede)
    self.rode_model = torch.load(self.path_rode)
    self.aede_model = torch.load(self.path_aede)

    # Dense Layers
    
    self.dense_fclayers = nn.Sequential(
#                                         nn.Dropout(0.5), 
                                        nn.BatchNorm1d(self.dense_size),
#                                         nn.Linear(self.dense_size, self.dense_size), nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(self.dense_size, self.hidden_size), nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(self.hidden_size, self.numclasses),
                                        )
    # Weigh initialization
    self.dense_fclayers.apply(self.cruiser_weights_init_uniform_rule)
    
    
    # Attention Layers

    q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.embeddim))
    self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(device)
    w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.embeddim, self.fchidden)) # noqa
    self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(device)


    # Conv Layers
    
    self.m11 = nn.Conv1d(in_channels=1, 
                  out_channels=self.conv_params["out_channels"], 
                  kernel_size=self.conv_params["kernel_size_1"], 
                  stride=self.conv_params["stride"], 
                  padding='valid')
    self.m12 = nn.Conv1d(in_channels=1, 
                  out_channels=self.conv_params["out_channels"], 
                  kernel_size=self.conv_params["kernel_size_2"], 
                  stride=self.conv_params["stride"], 
                  padding='valid')
    self.m21 = nn.Conv1d(in_channels=1, 
                  out_channels=self.conv_params["out_channels"], 
                  kernel_size=self.conv_params["kernel_size_1"], 
                  stride=self.conv_params["stride"], 
                  padding='valid')
    self.m22 = nn.Conv1d(in_channels=1, 
                  out_channels=self.conv_params["out_channels"], 
                  kernel_size=self.conv_params["kernel_size_2"], 
                  stride=self.conv_params["stride"], 
                  padding='valid')
    
    # Weight initialization
    self.m11.apply(self.conv_weights_init)
    self.m12.apply(self.conv_weights_init)
    self.m21.apply(self.conv_weights_init)
    self.m22.apply(self.conv_weights_init)
    
    # Pool Layers
    self.mp11 = nn.MaxPool1d(kernel_size=self.pool_params["pool_size_11"],)
    self.ap11 = nn.AvgPool1d(kernel_size=self.pool_params["pool_size_11"],)
    self.mp12 = nn.MaxPool1d(kernel_size=self.pool_params["pool_size_12"],)
    self.ap12 = nn.AvgPool1d(kernel_size=self.pool_params["pool_size_12"],)

    self.mp21 = nn.MaxPool1d(kernel_size=self.pool_params["pool_size_21"],)
    self.ap21 = nn.AvgPool1d(kernel_size=self.pool_params["pool_size_21"],)
    self.mp22 = nn.MaxPool1d(kernel_size=self.pool_params["pool_size_22"],)
    self.ap22 = nn.AvgPool1d(kernel_size=self.pool_params["pool_size_22"],)

  
  def forward(self, inp_ids, att_mask, token_ids):

    rode_output,hidden_states = self.rode_model(inp_ids, 
                                                att_mask,
                                                token_ids)
    _,aede_output = self.aede_model(rode_output)




    hidden_states = torch.stack([hidden_states[layer_i][:, 0].squeeze()
                                  for layer_i in range(0, self.numlayers)], dim=-1) # noqa
    hidden_states = hidden_states.view(-1, self.numlayers, self.embeddim)
    att_output = self.attention(hidden_states)
    
    # batch normalisation
#     att_output = self.at_norm(att_output) # attention outputs
#     aede_output = self.ae_norm(aede_output) # AE outputs 

    # batch normalisation
    att_output = self.dropout(att_output) # attention outputs
    aede_output = self.dropout(att_output) # AE outputs 

    input1 = torch.unsqueeze(att_output,1) # attention outputs
    input2 = torch.unsqueeze(aede_output,1) # AE outputs
    

    # attention out
    output11 = self.m11(input1)
    output12 = self.m12(input1)
    output_mp11 = self.mp11(output11).squeeze(2)
    output_ap11 = self.ap11(output11).squeeze(2)
    output_mp12 = self.mp12(output12).squeeze(2)
    output_ap12 = self.ap12(output12).squeeze(2)

    # denoiser out
    output21 = self.m21(input2)
    output22 = self.m22(input2)
    output_mp21 = self.mp21(output21).squeeze(2)
    output_ap21 = self.ap21(output21).squeeze(2)
    output_mp22 = self.mp22(output22).squeeze(2)
    output_ap22 = self.ap22(output22).squeeze(2)

    # concatenation
    output_cat1 = torch.cat((output_mp11, output_mp12, output_ap11, output_ap12), 1)
    output_cat2 = torch.cat((output_mp21, output_mp22, output_ap21, output_ap22), 1)
    output_catF = torch.cat((output_cat1, output_cat2), 1)
    
    # Dense Layers
    output = self.dense_fclayers(output_catF)

    return output

  def attention(self, h):
      v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
      v = F.softmax(v, -1)
      v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
      v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
      return v

  def cruiser_weights_init_uniform_rule(self, m):
      if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.normal_(m.bias, mean=0.0, std=1.0)
        
  def conv_weights_init(self, m):
      classname = m.__class__.__name__
      if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.normal_(m.bias, mean=0.0, std=1.0)      


def get_predictions(model, data_loader, mode, have_loss, device):
    model = model.eval()

    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for input_id, attention_masks, token_ids, labels in data_loader:
            
            input_ids = input_id.to(device)
            attention_mask = attention_masks.to(device)
            token_ids = token_ids.to(device)
            dummy_targets = labels.long().to(device)

            if mode != 'total':
              targets = torch.rand(dummy_targets.shape[0]).long().to(device)
            else:
              targets = dummy_targets

            if have_loss:

              _, outputs = model(input_ids,
                                attention_mask,
                                token_ids,
                                targets)
            else:

              outputs = model(input_ids,
                              attention_mask,
                              token_ids,)


            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(F.softmax(outputs, dim=-1))
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    
    return predictions, prediction_probs, real_values

 
def get_logits(model, test_loader, mode, have_loss, DEVICE):

  real_values = []

  #code to give the outputs
  with torch.no_grad():
      fin_outputs = []
      for input_id, attention_masks, token_ids, labels in test_loader:

          input_ids = input_id.to(DEVICE)
          attention_mask = attention_masks.to(DEVICE)
          token_ids = token_ids.to(DEVICE)
          targets = labels.long().to(DEVICE)

          if mode != 'total':
            dummy_targets = torch.rand(targets.shape[0]).long().to(DEVICE)
          else:
            dummy_targets = targets

          if have_loss:

            _, outputs = model(input_ids,
                              attention_mask,
                              token_ids,
                              dummy_targets)
          else:                    

            outputs = model(input_ids,
                            attention_mask,
                            token_ids,)

          outputs_np = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
          targets_np = targets.cpu().detach().numpy().tolist()
          fin_outputs.extend(outputs_np)
          real_values.extend(targets_np)

  fin_outputs_flatten = [item for sublist in fin_outputs for item in sublist]


  return fin_outputs_flatten, real_values


def merged_neutral(neuneg_probs, posneu_probs, size, threshold):

  tmp_neutral_results = torch.zeros(size=size)

  for idx, el in enumerate(tmp_neutral_results):
    if (neuneg_probs[idx][1] > threshold) and (posneu_probs[idx][0] > threshold):
      tmp_neutral_results[idx] = torch.tensor(1)  #Set neutral label

  return tmp_neutral_results

def merged_all(posneg_results, neutral_results, posneg_probs):

  all_results = neutral_results.clone()
  all_probs = torch.zeros(size=(posneg_results.shape[0], 3))

  for idx, el in enumerate(all_results):

    if all_results[idx] == torch.tensor(0):

      all_probs[idx][0] = posneg_probs[idx][0] # set negative prpbs
      all_probs[idx][2] = posneg_probs[idx][1] # set positive probs

      if (posneg_results[idx] == torch.tensor(1)):
        all_results[idx] = torch.tensor(2) # Set positive label
      else:
        all_results[idx] = torch.tensor(0) # set negative label

    else:
      all_probs[idx][0] = torch.tensor(0.1) 
      all_probs[idx][1] = torch.tensor(0.8) # set neutral probs
      all_probs[idx][2] = torch.tensor(0.1) 

  return all_results, all_probs

