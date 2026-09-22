import copy
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from MTP_Utils.mtp_model import Bert_Base, Bert_Attention, Bert_LSTM, Roberta_Base, Roberta_Attention
from MTP_Utils.mtp_model import Roberta_Denoiser, Roberta_Comp, AutoEncoder
from MTP_Utils.mtp_model import SentimentClassifier, SentimentCruiser
from MTP_Utils.dice_loss_rs import SelfAdjDiceLoss
from transformers import (get_linear_schedule_with_warmup, 
                          get_cosine_schedule_with_warmup, 
                          get_constant_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup, 
                          get_cosine_with_hard_restarts_schedule_with_warmup,)

# plt.rcParams["figure.figsize"] = (15,8)

# Log in to your W&B account
import wandb
from datetime import datetime
wandb.login(key="85d1febcb771fb3b2a08613be67a7f68c90bef03")

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
      
# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
#         n = m.in_features
#         y = 1.0/np.sqrt(n)
#         m.weight.data.uniform_(-y, y)
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu') # HE initialization
        try:
          torch.nn.init.normal_(m.bias, mean=0.0, std=1.0)
        except:
          pass
        
def cruiser_weights_init_uniform_rule(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.normal_(m.bias, mean=0.0, std=1.0)
    

def evaluate(loader, net, net_aede, net_comp, denoising, device, model_name):
  """ Evaluates a model and returns loss, accuracy

  Arguments:
  loader (DataLoader): dataloader to evaluate
  net (nn.Module): Model to evaluate
  device (torch.device): Device type
  model_name (str): Model name

  """
  net.eval()

  if net_aede is not None:
    net_aede.eval()
    net_comp.eval()

  with torch.no_grad():
    loss = 0.0
    total = 0.0
    acc = 0.0
    y_pred = []
    y_true = []

    if denoising:
      criterion = torch.nn.MSELoss()
#       criterion = torch.nn.HuberLoss()

      for input_id, attention_masks, token_ids, input_id_com, attention_masks_com, token_ids_com, labels in loader:

        input_id = input_id.to(device)
        attention_masks = attention_masks.to(device)
        token_ids = token_ids.to(device)

        input_id_com = input_id_com.to(device)
        attention_masks_com = attention_masks_com.to(device)
        token_ids_com = token_ids_com.to(device)

        labels = labels.long().to(device)

        if(model_name == 'base'):
          curloss, output = net(input_id, attention_masks, token_ids, labels)# noqa
        elif(model_name == 'roberta'):
          curloss, output = net(input_id, attention_masks, token_ids, labels)# noqa
        elif(model_name == 'roberta_denoiser'):
          output_rode,_ = net(input_id, attention_masks, token_ids)
          _,output_aede = net_aede(output_rode)
          output_comp = net_comp(input_id_com, attention_masks_com, token_ids_com)
          curloss = criterion(output_aede, output_comp)


        else:
          output = net(input_id, attention_masks, token_ids)
          curloss = F.cross_entropy(output, labels, reduction='mean')
            
        loss += curloss.item()
        total += input_id.size(0)

      return round((loss / total), 6)

    else:
      for input_id, attention_masks, token_ids, labels in loader:
        input_id = input_id.to(device)
        attention_masks = attention_masks.to(device)
        token_ids = token_ids.to(device)
        labels = labels.long().to(device)

        if(model_name == 'base'):
          curloss, output = net(input_id, attention_masks, token_ids, labels)# noqa
        elif(model_name == 'roberta'):
          curloss, output = net(input_id, attention_masks, token_ids, labels)# noqa
        else:
          output = net(input_id, attention_masks, token_ids)
          curloss = F.cross_entropy(output, labels, reduction='mean')
        loss += curloss.item()
        preds = torch.argmax(output, 1)
        y_pred.extend(preds.tolist())
        y_true.extend(labels.tolist())
        acc += torch.sum(preds == labels).item()
        total += input_id.size(0)

      F1 = round((f1_score(y_true, y_pred, average='macro')), 2) * 100
      return round((loss / total), 3), round(((acc / total) * 100), 2), F1


def train_model(train_loader, dev_loader, test_loader,
                model_name, denoising,
                numclasses, numepochs,
                runs, device, 
                hy_args,
                check_name, 
                rode_model_path, aede_model_path):
  """ Trains the neural network

  Arguments:
  train_loader (DataLoader): Training Data Loader
  dev_loader (DataLoader): Validation Data Loader
  test_loader (DataLoader): Test Data Loader
  model_name (str): Name of the model to train
  numclasses (int): Number of classes in the data
  numepochs (int): Number of epochs to train
  runs (int): Number of runs to report averaged results
  device (torch.device): Device type
  check_name (str) : Name of the model so save checkpoint

  """
  avg_testacc = 0.0
  avg_testf1 = 0.0
  best_acc = 0.0
  best_loss = 0.0 + 1e10
  avg_testloss = 0.0
  total_steps = len(train_loader) * numepochs
  num_iters = len(train_loader)
  dice_loss = SelfAdjDiceLoss()
  class_weights = torch.FloatTensor([2.0, 4.0, 1.0]).to(device)

  for run in range(1, runs+1):
    print("Training for run {} ".format(run))
    print("--------------------------------------------")
    if(model_name == 'lstm'):
      model = Bert_LSTM(numclasses).to(device)
    elif(model_name == 'bert_attention'):
      model = Bert_Attention(numclasses, device).to(device)
    elif(model_name == 'roberta_attention'):
      model = Roberta_Attention(numclasses, device).to(device)
    elif(model_name == 'roberta_denoiser'):
      model_rode = Roberta_Denoiser(numclasses, device, rode_model_path).to(device)
      model_aede = AutoEncoder(device).to(device)
      model_comp = Roberta_Comp(numclasses, device, rode_model_path).to(device)
      
      # applies the specified weight initialization
      model_aede.apply(weights_init_uniform_rule)
      
    elif(model_name =='roberta'):
      model = Roberta_Base(numclasses).to(device)
    elif(model_name == 'roberta_de_class'):
      model = SentimentClassifier(numclasses,
                                  rode_model_path=rode_model_path, 
                                  aede_model_path=aede_model_path).to(device)
      # Weight initialization
#       model.apply(weights_init_uniform_rule)
                                       
    elif(model_name == 'roberta_de_class_cruiser'):
      model = SentimentCruiser(numclasses, device,
                               rode_model_path=rode_model_path, 
                               aede_model_path=aede_model_path).to(device)
      # Weight initialization
#       model.apply(cruiser_weights_init_uniform_rule)
    else:
      model = Bert_Base(numclasses).to(device)

    if denoising:
      optimizer = torch.optim.Adam(model_aede.parameters(),
                                   lr=hy_args["denoising_lr"],
                                   weight_decay=hy_args["denoising_wdecay"])

#       scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
#                                                                      num_warmup_steps=200,
#                                                                      num_training_steps=total_steps,
#                                                                      num_cycles=5)

      scheduler = get_linear_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=int(total_steps / 5),
                                                  num_training_steps=total_steps,
                                                   )

    else:
      optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=hy_args["classifying_lr"], 
                                    weight_decay=hy_args["classifying_wdecay"])

      scheduler = get_linear_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=int(total_steps * 0.05),
                                                  num_training_steps=total_steps)

    valbest = 0.0
    lossbest = 0.0 + 1e10
    history_denoising = defaultdict(list)

    
    if denoising:
      model_rode.train()
      model_aede.train()
      model_comp.train()
      best_model_rode_wts = copy.deepcopy(model_rode.state_dict())
      best_model_aede_wts = copy.deepcopy(model_aede.state_dict())

      criterion = torch.nn.MSELoss()
#       criterion = torch.nn.HuberLoss()

      project_name = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
      wandb.init(
        # Set the project where this run will be logged
        project=f"acl14_denoising_{project_name}", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{run}",
        )
      print(f"Project Name:  acl14_denoising_{project_name}")
      wandb.config = hy_args

      for epoch in range(1, numepochs+1):
        model_rode.train()
        model_aede.train()
        model_comp.train()
        sch_counter = 0

        for input_id, attention_masks, token_ids, input_id_com, attention_masks_com, token_ids_com, labels in train_loader:

          input_id = input_id.to(device)
          attention_masks = attention_masks.to(device)
          token_ids = token_ids.to(device)

          input_id_com = input_id_com.to(device)
          attention_masks_com = attention_masks_com.to(device)
          token_ids_com = token_ids_com.to(device)                
          
          labels = labels.long().to(device)

          model_rode.zero_grad()
          model_aede.zero_grad()
          model_comp.zero_grad()

          if(model_name == 'base'):
            loss, _ = model(input_id, attention_masks, token_ids, labels) # noqa
          elif(model_name == 'roberta'):
            loss, _ = model(input_id, attention_masks, token_ids, labels) # noqa

          elif(model_name == 'roberta_denoiser'):
            output_rode,_ = model_rode(input_id, attention_masks, token_ids)
            _,output_aede = model_aede(output_rode)
            output_comp = model_comp(input_id_com, attention_masks_com, token_ids_com)

            loss = criterion(output_aede, output_comp)

          else:
            output = model(input_id, attention_masks, token_ids)
            loss = F.cross_entropy(output, labels)
            # loss = dice_loss(output, labels)

          loss.backward()
          nn.utils.clip_grad_norm_(model_aede.parameters(), max_norm=1.0)
          optimizer.step()          
          scheduler.step()
          sch_counter += 1

        history_denoising['train_loss'].append(loss.item())

        valloss = evaluate(dev_loader, 
                           model_rode,
                           model_aede, 
                           model_comp,
                           denoising, 
                           device, 
                           model_name) # noqa
        history_denoising['val_loss'].append(valloss)
        
        if(valloss < lossbest):
            lossbest = valloss
            best_model_rode_wts = copy.deepcopy(model_rode.state_dict())
            best_model_aede_wts = copy.deepcopy(model_aede.state_dict())

        print("Epoch {} |Train Loss {:.6f} |Val Loss {} ".format(epoch, loss.item(), valloss))
        wandb.log({"Train Loss": loss.item(), "Val Loss": valloss})
        
    else:
      model.train()
      best_model_wts = copy.deepcopy(model.state_dict())

      for epoch in range(1, numepochs+1):
        model.train()
        total = 0.0
        acc = 0.0
        y_pred = []
        y_true = []
    
        for input_id, attention_masks, token_ids, labels in train_loader:
          input_id = input_id.to(device)
          attention_masks = attention_masks.to(device)
          token_ids = token_ids.to(device)
          labels = labels.long().to(device)

          model.zero_grad()

          if(model_name == 'base'):
            loss, _ = model(input_id, attention_masks, token_ids, labels) # noqa
          elif(model_name == 'roberta'):
            loss, _ = model(input_id, attention_masks, token_ids, labels) # noqa

          else:
            output = model(input_id, attention_masks, token_ids)
            loss = F.cross_entropy(output, labels)
            # loss = dice_loss(output, labels)

          loss.backward()
          nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()
          scheduler.step()
            
          with torch.no_grad():
            preds = torch.argmax(output, 1)
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())
            acc += torch.sum(preds == labels).item()
            total += input_id.size(0)
            
        model_comp = None
        model_aede = None
        
        trainacc = round(((acc / total) * 100), 2)

        valloss, valacc, _ = evaluate(dev_loader, 
                                      model,
                                      model_comp,
                                      model_aede,
                                      denoising,
                                      device, 
                                      model_name) # noqa
        if(valacc > valbest):
            valbest = valacc
            best_model_wts = copy.deepcopy(model.state_dict())

        print("Epoch {}  Train Acc {} Val Loss {} Val Acc {} ".format(epoch,
                                                                      trainacc,
                                                                      valloss,
                                                                      valacc))

    if denoising:
      model_rode.load_state_dict(best_model_rode_wts)
      model_aede.load_state_dict(best_model_aede_wts)


      curtestloss = evaluate(test_loader, 
                             model_rode,
                             model_aede, 
                             model_comp,
                             denoising, 
                             device, 
                             model_name)

      print("Run {} Test Loss {} ".format(run, curtestloss))
      print("---------------------------------------------------")
      plt.plot(history_denoising['train_loss'], label='Denoising train loss')
      plt.plot(history_denoising['val_loss'], label='Denoising validation loss')
      plt.title('Denoising Loss history')
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.legend()
      plt.ylim([0, max(history_denoising['train_loss'])]);
      plt.show()
      print("---------------------------------------------------")
      wandb.finish()

      if curtestloss < best_loss:
        model_rode_path = 'MTP_Utils/checkpoints/roberta_denoiser_{}.pt'.format(check_name)
        model_aede_path = 'MTP_Utils/checkpoints/autoenc_denoiser_{}.pt'.format(check_name)

        torch.save(model_rode, model_rode_path)
        torch.save(model_aede, model_aede_path)

        best_loss = curtestloss
        
      avg_testloss += curtestloss


    else:
      model.load_state_dict(best_model_wts)
      model_comp = None
      model_aede = None
      curtestloss, curtestacc, curtestf1 = evaluate(test_loader, 
                                                    model, 
                                                    model_comp,
                                                    model_aede,
                                                    denoising,
                                                    device, 
                                                    model_name)

      print("Run {} Test Accuracy {} F1 Score {}".format(run,
                                                        curtestacc,
                                                        curtestf1))
      print("---------------------------------------------------")

      if curtestacc > best_acc:
        model_path = 'MTP_Utils/checkpoints/bert_{}.pt'.format(check_name)
        torch.save(model, model_path)
        best_acc = curtestacc
        
      avg_testacc += curtestacc
      avg_testf1 += curtestf1
        

  if denoising:
    print("Average Test Loss: {} ".format(avg_testloss/runs))
    return (model_rode_path, model_aede_path)

  else:
    print("Average Test Accuracy: {} F1: {} ".format(avg_testacc/runs,
                                                    avg_testf1/runs))
    return model_path
