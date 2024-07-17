### Imports ###

import torch
from transformers import AutoTokenizer, AutoModel

### End of Imports


### Function Definition ###

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):

    # Get token embeddings from the output of a model. First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Takes a list containing sentence(s) and encodes them to tokens, them creates vectors representing sentence features.
def embedSentences(sentences: list,
                   tokenizer='sentence-transformers/all-MiniLM-L6-v2', # Name of tokenizer OR path to local file.
                   model='sentence-transformers/all-MiniLM-L6-v2', # Name of model OR path to local file.
                   use_local=False, # Whether to try finding the model on HuggingFace or load from local location.
                   padding=True,
                   truncation=True, # Whether to truncate token sequence above input length of 256.
                   max_length=256 #Default max length of token sequence to consider.
                   ):

    # Load AutoModel from huggingface model repository
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=use_local)
    model = AutoModel.from_pretrained(model, local_files_only=use_local)

    # Tokenize sentences.
    encoded_input = tokenizer(sentences,
                              padding=padding,
                              truncation=truncation,
                              max_length=max_length,
                              return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean(average) pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Having converted sentences to a numerical representation of their semantics, return the vectors for each sentence.
    return sentence_embeddings

### End of Function Definition ###


### Frozen Variables ###

# Sentences we want sentence embeddings for
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.',
             'The quick brown cat jumps over the lazy beaver']

### End of Frozen Variables ###

### Dynamic Logic ###

# Encodes sentences to tokens, encodes tokens to a vector embedding for clustering and semantic similarity.
encodedSentences = embedSentences(sentences, use_local=False)

### End of Dynamic Logic ###


























