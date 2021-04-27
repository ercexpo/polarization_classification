from sklearn.metrics import f1_score
from bert_document_classification.document_bert import BertForDocumentClassification
from bert_document_classification.document_bert import encode_documents
from pprint import pformat
import load_data
import time, logging, torch, configargparse, os, socket

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#load comments and labels from the input tsv
comments, labels = load_data.get_data(sys.argv[1])
val_comments, val_labels = load_data.get_data(sys.argv[2])

#encode inputs using BERT tokenizer
input_ids = []
val_input_ids = []

for comment in comments:
    encoded_comment = tokenizer.encode(comment, add_special_tokens = True, max_length=512,pad_to_max_length=True,truncation=True)
    input_ids.append(encoded_comment)

for comment in val_comments:
    encoded_comment = tokenizer.encode(comment, add_special_tokens = True, max_length=512,pad_to_max_length=True,truncation=True)
    val_input_ids.append(encoded_comment)


#define attention masks: if 0 it's a PAD, set to 0; else set to 1
attention_masks = []
val_attention_masks = []

for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

for sent in val_input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    val_attention_masks.append(att_mask)

labels = labels.astype(np.int)

val_labels = val_labels.astype(np.int)

#train_test_val split
#train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
#                                                            random_state=2018, test_size=0.1)

#train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
#                                             random_state=2018, test_size=0.1)

# Convert all inputs and labels into torch tensors, the required datatype
# for our model.
train_inputs = torch.tensor(input_ids)
validation_inputs = torch.tensor(val_input_ids)
                                                           
