import torch
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import AutoTokenizer


def get_pretrained_tokenizer(model_name):
    """
    Loads pretrained Tokenizer and returns it,
    """
    print("Downloading tokenizer to cache")
    print("---------------------------------------")

    if model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  do_lower_case=False)
    else:
        print("You should determine model name")

    return tokenizer


def tokenize_sentences(bert_tokenizer, sentences, aspects, maxlen):
    """ converts sentences into ids according to bert tokenizer
    Updated:21:19

    Arguments:
    bert_tokenizer (Tokenizer): Pretrained BERT Tokenizer
    sentences (list): List of sentences
    maxlen (int): Maximum Length of a sentence
    
    Outputs:
    input_ids
    input_attention_masks
    input_type_ids

    """
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for sentence, aspect in zip(sentences, aspects):

        encoded = bert_tokenizer.encode_plus(text=sentence,
                                             text_pair=aspect,
                                             add_special_tokens=True,
                                             max_length=maxlen,
                                             return_token_type_ids=True,
                                             padding='max_length',
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_tensors='pt')
        
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        token_type_ids.append(encoded['token_type_ids'])

    input_ids = torch.cat(input_ids, dim=0, out=None)
    attention_masks = torch.cat(attention_masks, dim=0, out=None)
    token_type_ids = torch.cat(token_type_ids, dim=0, out=None)

    return input_ids, attention_masks, token_type_ids