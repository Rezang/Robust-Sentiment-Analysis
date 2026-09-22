import torch
import pandas as pd
import numpy as np
from MTP_Utils.mtp_tokenizer import get_pretrained_tokenizer, tokenize_sentences
from MTP_Utils.loader import get_loader, get_loader_joint



def tok_loader(df_train, df_val, df_test,
               maxlen, batch_size, have_aspect, 
               txt_col, com_col, sen_col, 
               model_name):

    MAX_LEN = maxlen
    BATCH_SIZE = batch_size


    train_sentence = df_train[txt_col].to_list()
    val_sentence = df_val[txt_col].to_list()
    test_sentence = df_test[txt_col].to_list()

    if com_col:
      train_com_sentence = df_train[com_col].to_list()
      val_com_sentence = df_val[com_col].to_list()
      test_com_sentence = df_test[com_col].to_list()  

    if have_aspect:
      train_aspect = df_train.aspects.to_list()
      val_aspect = df_val.aspects.to_list()
      test_aspect = df_test.aspects.to_list()
    else:
      train_aspect = [None for s in train_sentence]
      val_aspect = [None for s in val_sentence]
      test_aspect = [None for s in test_sentence]

    train_label = df_train[sen_col].to_list()
    val_label = df_val[sen_col].to_list()
    test_label = df_test[sen_col].to_list()

    train_labels = torch.from_numpy(np.asarray(train_label, 'int32'))
    val_labels = torch.from_numpy(np.asarray(val_label, 'int32'))
    test_labels = torch.from_numpy(np.asarray(test_label, 'int32'))


    (train_input_ids,
     train_attention_masks,
     train_type_ids) = tokenize_sentences(get_pretrained_tokenizer(model_name),
                                          train_sentence,
                                          train_aspect,
                                          maxlen=MAX_LEN)
    (val_input_ids,
     val_attention_masks,
     val_type_ids) = tokenize_sentences(get_pretrained_tokenizer(model_name),
                                        val_sentence,
                                        val_aspect,
                                        maxlen=MAX_LEN)
    (test_input_ids,
     test_attention_masks,
     test_type_ids) = tokenize_sentences(get_pretrained_tokenizer(model_name),
                                         test_sentence,
                                         test_aspect,
                                         maxlen=MAX_LEN)

    # Tokenize complete sentences
    if com_col:

      (train_input_ids_com,
       train_attention_masks_com,
       train_type_ids_com) = tokenize_sentences(get_pretrained_tokenizer(model_name),
                                                train_com_sentence,
                                                train_aspect,
                                                maxlen=MAX_LEN)
      (val_input_ids_com,
       val_attention_masks_com,
       val_type_ids_com) = tokenize_sentences(get_pretrained_tokenizer(model_name),
                                              val_com_sentence,
                                              val_aspect,
                                              maxlen=MAX_LEN)
      (test_input_ids_com,
       test_attention_masks_com,
       test_type_ids_com) = tokenize_sentences(get_pretrained_tokenizer(model_name),
                                               test_com_sentence,
                                               test_aspect,
                                               maxlen=MAX_LEN)
                                         
    print("Tokenizing Completed")


    if com_col:

      train_loader = get_loader_joint(train_input_ids,
                                      train_attention_masks,
                                      train_type_ids,
                                      train_input_ids_com,
                                      train_attention_masks_com,
                                      train_type_ids_com,
                                      train_labels.long(),
                                      batchsize=BATCH_SIZE)

      val_loader = get_loader_joint(val_input_ids,
                                    val_attention_masks,
                                    val_type_ids,
                                    val_input_ids_com,
                                    val_attention_masks_com,
                                    val_type_ids_com,
                                    val_labels.long(),
                                    batchsize=BATCH_SIZE)

      test_loader = get_loader_joint(test_input_ids,
                                     test_attention_masks,
                                     test_type_ids,
                                     test_input_ids_com,
                                     test_attention_masks_com,
                                     test_type_ids_com,
                                     test_labels.long(),
                                     batchsize=BATCH_SIZE) 

    else:
      train_loader = get_loader(train_input_ids,
                                train_attention_masks,
                                train_type_ids,
                                train_labels.long(),
                                batchsize=BATCH_SIZE)

      val_loader = get_loader(val_input_ids,
                              val_attention_masks,
                              val_type_ids,
                              val_labels.long(),
                              batchsize=BATCH_SIZE)

      test_loader = get_loader(test_input_ids,
                              test_attention_masks,
                              test_type_ids,
                              test_labels.long(),
                              batchsize=BATCH_SIZE)


   

    print("DataLoader Created\n")
    
    print(" 😀 😀 😀 😀 😀 \n")                                         


    return train_loader, val_loader, test_loader
    
