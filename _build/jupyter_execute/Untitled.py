#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# choose between 'local-tgz' and 'web',
# see description under each heading below

data_source = 'local-tgz'

# adjust path as needed

data_tgz = os.path.join('data', 'pang-and-lee-2004', 'data.tar.gz')


# In[2]:


#from collections import defaultdict

class PL04DataLoader_Part_1:
    
    def __init__(self):
        pass
    
    def get_labelled_dataset(self, fold = 0):
        ''' Compile a fold of the data set
        '''
        dataset = []
        for label in ('pos', 'neg'):
            for document in self.get_documents(
                fold = fold,
                label = label,
            ):
                dataset.append((document, label))
        return dataset
    
    def get_documents(self, fold = 0, label = 'pos'):
        ''' Enumerate the raw contents of selected data set files.
            Args:
                data_dir: relative or absolute path to the data set folder
                fold: which fold to load (0 to n_folds-1)
                label: 'pos' or 'neg' to
                    select data with positive or negative sentiment
                    polarity
            Return:
                List of tokenised documents, each a list of sentences
                that in turn are lists of tokens
        '''
        raise NotImplementedError


# In[3]:


class PL04DataLoader(PL04DataLoader_Part_1):
    
    def get_xval_splits(self):
        ''' Split data with labels for cross-validation
            returns a list of k pairs (training_data, test_data)
            for k cross-validation
        '''
        # load the folds
        folds = []
        for i in range(10):
            folds.append(self.get_labelled_dataset(
                fold = i
            ))
        # create training-test splits
        retval = []
        for i in range(10):
            test_data = folds[i]
            training_data = []
            for j in range(9):
                ij1 = (i+j+1) % 10
                assert ij1 != i
                training_data = training_data + folds[ij1]
            retval.append((training_data, test_data))
        return retval


# In[4]:


import tarfile
import time

class PL04DataLoaderFromStream(PL04DataLoader):
        
    def __init__(self, tgz_stream, **kwargs):
        super().__init__(**kwargs)
        self.data = {}
        counter = 0
        with tarfile.open(
            mode = 'r|gz',
            fileobj = tgz_stream
        ) as tar_archive:
            for tar_member in tar_archive:
                if counter == 2000:
                    break
                path_components = tar_member.name.split('/')
                filename = path_components[-1]
                if filename.startswith('cv')                 and filename.endswith('.txt')                 and '_' in filename:
                    label = path_components[-2]
                    fold = int(filename[2])
                    key = (fold, label)
                    if key not in self.data:
                        self.data[key] = []
                    f = tar_archive.extractfile(tar_member)
                    document = [
                        line.decode('utf-8').split()
                        for line in f.readlines()
                    ]
                    self.data[key].append(document)
                    counter += 1
            
    def get_documents(self, fold = 0, label = 'pos'):
        return self.data[(fold, label)]


# In[5]:


class PL04DataLoaderFromTGZ(PL04DataLoaderFromStream):
    
    def __init__(self, data_path, **kwargs):
        with open(data_path, 'rb') as tgz_stream:
            super().__init__(tgz_stream, **kwargs)


# In[6]:


# adjust the torch version below following instructions on https://pytorch.org/get-started/locally/

import sys

# for why we use {sys.executable} see
# https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/

try:
    import torch
except ModuleNotFoundError:
    get_ipython().system('{sys.executable} -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html')

try:
    import transformers
except ModuleNotFoundError:
    get_ipython().system('{sys.executable} -m pip install transformers')

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    get_ipython().system('{sys.executable} -m pip install pytorch-lightning')

try:
    import torchnlp
except ModuleNotFoundError:
    get_ipython().system('{sys.executable} -m pip install pytorch-nlp')

try:
    import tensorboard
except ModuleNotFoundError:
    get_ipython().system('{sys.executable} -m pip install tensorboard')

print('Ready')


# In[7]:


from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Whitespace

model_size = 'base'  # choose between 'tiny', 'base' and 'large'

size2name = {
    'tiny':  'distilbert-base-uncased',  # TODO: This doesn't seem to reduce memory usage compared to bert-base. 
    'base':  'bert-base-uncased',
    'large': 'bert-large-uncased',
}


model_name = size2name[model_size]

force_whitespace_pre_tokeniser = False  # should not matter with pre-tokenised P&L'04 input

tokeniser = AutoTokenizer.from_pretrained(model_name)
    
if force_whitespace_pre_tokeniser:
    tokeniser.pre_tokenizer = Whitespace()


# In[8]:


# let's test the tokeniser with a small example

example_batch = [
    'hello world !'.split(),
    """tokenisation 's fun""".split(),
]

tokenised_text = tokeniser(
    example_batch,  # pre-tokenised input
    is_split_into_words = True,
)

for i, token_ids in enumerate(tokenised_text['input_ids']):
    print(i, "\t['input_ids']:", token_ids)
    print(   '\ttokens:       ', tokeniser.convert_ids_to_tokens(token_ids))
    print(   "\t.word_ids():  ", tokenised_text.word_ids(batch_index = i))
    example = example_batch[i]
    for token in example:
        # https://stackoverflow.com/questions/62317723/tokens-to-words-mapping-in-the-tokenizer-decode-step-huggingface
        print('%18s --> %r' %(
            token,
            tokeniser.encode(
                token,
                add_special_tokens = False,
            )
        ))


# In[9]:


# if you have sufficient TPU or GPU memory you can increase the sequence length up to 512
# (memory requirements will be the highest during fine-tuning of BERT)
max_sequence_length  = 256

# memory requirements increase linearly with the batch size;
# a batch size of 16, better 32, is needed for efficient training;
# however, with little memory, we have to go lower
batch_size      = 10   # for a 12 GB card


# In[11]:


from collections import defaultdict
    
bin_width = 250
data_loader = PL04DataLoaderFromTGZ('data.tar.gz')
distribution = defaultdict(lambda: 0)

# interate all reviews
for fold in range(10):
    for label in 'pos neg'.split():
        batch = []
        for document in data_loader.get_documents(
            label = label,
            fold = fold,
        ):
            tokens = []
            for sentence in document:
                for token in sentence:
                    tokens.append(token)
            batch.append(tokens)
        tokenised_batch = tokeniser(
            batch,  # pre-tokenised input
            is_split_into_words = True,
        )
        max_length_bin = 0
        for token_ids in tokenised_batch['input_ids']:
            length = len(token_ids)
            length_bin = length // bin_width
            distribution[(label,   length_bin)] += 1
            distribution[('total', length_bin)] += 1
            if length_bin > max_length_bin:
                max_length_bin = length_bin
print('LengthBin\tPos\tNeg\tTotal')
for length_bin in range(0, max_length_bin+1):
    row = []
    row.append('%4d-%4d' %(
        bin_width*length_bin,
        bin_width*(1+length_bin)-1
    ))
    for label in 'pos neg total'.split():
        count = distribution[(label, length_bin)]
        row.append('%d' %count)        
    print('\t'.join(row))
            


# In[12]:


# basic usage of pytorch and lightning from
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# and
# https://github.com/ricardorei/lightning-text-classification/blob/master/classifier.py
    
from torch.utils.data import Dataset, DataLoader, RandomSampler

class SlicedDocuments_part_1(Dataset):
    
    def __init__(
        self,
        raw_data,
        tokeniser = None,
        fraction_for_first_sequence = 0.5,
        max_sequence_length = 128,
        second_part_as_sequence_B = False,
        preproc_batch_size = 8,
        
    ):
        '''Extracts slices from labelled documents
           Args:
               raw_data: list of (document, label) pairs
               tokeniser: transformer tokeniser to obtain subword units; this tokeniser
                   will be used to decide how many words to include in each slice; it
                   is recommended to use the same tokeniser that will be used to tokenise
                   the data
               fraction_for_first_sequence: 1 = take slice from the start of the document,
                   0 = take slice from the end of the document, > 0 and < 1 = take two
                   slices, one of this relative size from the start and then fill to
                   max_sequence_length from the end of the document
               max_sequence_length: produce sequences up to this length
               allow_partial_tokens: whether to slice between subword units of tokens; if
                   set to True the produced sequences will always have max_sequence_length
                   items unless the document is very short
        '''
        assert max_sequence_length >= 5
        self.max_sequence_length = max_sequence_length
        self.second_part_as_sequence_B = second_part_as_sequence_B
        self.init_sequence_lengths(fraction_for_first_sequence)
        self.tokeniser = tokeniser
        self.init_slices(raw_data, preproc_batch_size)

    def init_sequence_lengths(self, fraction_for_first_sequence):
        available_for_two_sequences = self.max_sequence_length - 3
        self.first_sequence_length = max(0, int(
            fraction_for_first_sequence * available_for_two_sequences
        ))
        self.last_sequence_length = max(
            0,
            available_for_two_sequences - self.first_sequence_length
        )
        assert self.first_sequence_length             + self.last_sequence_length               <= available_for_two_sequences
        if self.first_sequence_length == 0:
            self.last_sequence_length += 1
        elif self.last_sequence_length == 0:
            self.first_sequence_length += 1
        if self.first_sequence_length == 0         or self.last_sequence_length == 0:
            # do not use a second [SEP] marker when there is
            # always only one sequence
            self.second_part_as_sequence_B = False


# In[13]:


class SlicedDocuments_part_2(SlicedDocuments_part_1):
    
    def init_slices(self, raw_data, preproc_batch_size):
        self.slices = []
        next_batch = []
        next_labels = []
        for document, label in raw_data:
            tokens = []
            for sentence in document:
                tokens = tokens + sentence
            next_batch.append(tokens)    
            next_labels.append(label)
            if len(next_batch) >= preproc_batch_size:
                self.add_batch(next_batch, next_labels)
                next_batch = []
                next_labels = []
        if next_batch:
            self.add_batch(next_batch, next_labels)


# In[14]:


class SlicedDocuments_part_3(SlicedDocuments_part_2):
    
    def add_batch(self, batch, labels):
        # determine, for each document in the batch, how many
        # tokens to include from the start of the document
        if self.first_sequence_length:
            lengths_1 = self.get_lengths(batch)
        else:
            lengths_1 = len(batch) * [0]
        # determine, for each document in the batch, how many
        # tokens to include from the end of the document    
        if self.last_sequence_length:
            lengths_2 = self.get_lengths(batch, part = 2, lengths_1 = lengths_1)
        else:
            lengths_2 = len(batch) * [0]
        # TODO: In an earlier version, we did not check the following
        #       condition, creating a second sequence for short
        #       documents even though only one sequence is requested.
        #       First results indicate that this bug actually
        #       improves performance. Future work should investigate
        #       this and, if this effect is confirmed, propose a
        #       clean solution to exploit this effect.
        if self.first_sequence_length:
            # sometimes there is space for more tokens from the
            # start even though no more from the end fit
            lengths_1 = self.expand_lengths(batch, lengths_1, lengths_2)
        # TODO: For cases with length_1 + length_2 > len(tokens),
        #       should we adjust the parts to not overlap?       
        # prepare texts
        for batch_idx, tokens in enumerate(batch):
            parts = []
            length_1 = lengths_1[batch_idx]
            length_2 = lengths_2[batch_idx]
            part_1 = tokens[:length_1]
            if length_2 > 0:
                part_2 = tokens[-length_2:]
            else:
                part_2 = []
            if self.second_part_as_sequence_B:
                parts.append(part_1)
                parts.append(part_2)
            else:
                parts.append(part_1 + part_2)
            assert len(parts) > 0    
            self.slices.append((parts, labels[batch_idx]))


# In[15]:


class SlicedDocuments_part_4(SlicedDocuments_part_3):

    def get_lengths(
        self, batch, part = 1, lengths_1 = None,
        lengths_2 = None,
    ):
        if part == 3:
            lower_limits = lengths_1[:]  # clone
        else:
            lower_limits = len(batch) * [0]
        upper_limits = []
        for tokens in batch:
            upper_limits.append(min(
                len(tokens),
                self.max_sequence_length
            ))
        # we want each upper limit to be a sequence length
        # that is too big but sometimes the full document
        # (or max_length words) can fit --> test for this
        # special case
        for batch_idx, fit in enumerate(
            self.get_fit(batch, upper_limits, part, lengths_1, lengths_2)
        ):
            if fit:
                # update lower limit to match upper limit
                # to mark this document as not needing any
                # further length search
                lower_limits[batch_idx] = upper_limits[batch_idx]
        while True:
            # prepare next lengths to test and check whether search is finished
            new_limits = []
            all_done = True
            for batch_idx, tokens in enumerate(batch):
                if lower_limits[batch_idx]+1 >= upper_limits[batch_idx]:
                    new_limits.append(lower_limits[batch_idx])
                else:
                    all_done = False
                    new_limits.append((
                        lower_limits[batch_idx] + upper_limits[batch_idx]
                    )//2)
            if all_done:
                return lower_limits
            # adjust lower and upper limits
            for batch_idx, fit in enumerate(
                self.get_fit(batch, new_limits, part, lengths_1, lengths_2)
            ):
                if fit:
                    lower_limits[batch_idx] = new_limits[batch_idx]
                else:
                    upper_limits[batch_idx] = new_limits[batch_idx]


# In[16]:


class SlicedDocuments_part_5(SlicedDocuments_part_4):

    def get_fit(self, batch, limits, part, lengths_1, lengths_2 = None):
        sliced_batch_A = []
        sliced_batch_B = []
        for batch_idx, tokens in enumerate(batch):
            if part == 1:
                length_1 = limits[batch_idx]
                length_2 = 0
            elif part == 2:
                length_1 = lengths_1[batch_idx]
                length_2 = limits[batch_idx]
            else:
                length_1 = limits[batch_idx]
                length_2 = lengths_2[batch_idx]
            part_1 = tokens[:length_1]
            if length_2 > 0:
                part_2 = tokens[-length_2:]
            else:
                part_2 = []
            if self.second_part_as_sequence_B:
                sliced_batch_A.append(part_1)
                sliced_batch_B.append(part_2)
            else:
                sliced_batch_A.append(part_1 + part_2)
        if self.second_part_as_sequence_B:
            tokenised = self.tokeniser(
               sliced_batch_A, sliced_batch_B,
               is_split_into_words = True,
            )
        else:
            tokenised = self.tokeniser(
               sliced_batch_A,
               is_split_into_words = True,
            )
        # check lengths in subword pieced
        retval = []
        for batch_idx, subword_ids in enumerate(tokenised['input_ids']):
            if part == 1:
                length = len(subword_ids) - 2  # account for [CLS] and [SEP]
                if self.second_part_as_sequence_B:
                    length -= 1                # account for second [SEP] token
                fit = length <= self.first_sequence_length
            else:
                fit = len(subword_ids) <= self.max_sequence_length
            retval.append(fit)
        return retval


# In[17]:


class SlicedDocuments(SlicedDocuments_part_5):

    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            assert isinstance(idx, int)
        parts, label = self.slices[idx]
        retval = {}
        retval['parts'] = parts
        retval['label'] = label
        return retval
    
    def expand_lengths(self, batch, lengths_1, lengths_2):
        ''' pushes lengths_1 as far out as possible '''
        return self.get_lengths(
            batch, part = 3,
            lengths_1 = lengths_1,
            lengths_2 = lengths_2,
        )


# In[18]:


splits = data_loader.get_xval_splits()
print('Ready')


# In[19]:


# this may take a minute

raw_data = splits[0][0]
sliced_docs = SlicedDocuments(
    raw_data, tokeniser,
    fraction_for_first_sequence = 0.25,
    max_sequence_length = max_sequence_length, # 99,
    second_part_as_sequence_B = True,
)
print('%d training documents ready in x-val fold 0' %len(sliced_docs))


# In[20]:


length2count = defaultdict(lambda: 0)

print('doc_idx seq_len tokens1 tokens2 total')
for doc_idx, sliced_doc in enumerate(sliced_docs):
    parts = sliced_doc['parts']
    if len(parts) == 2:
        tokenised = tokeniser(
           [parts[0]],
           [parts[1]],
           is_split_into_words = True,
        )
        tokens1 = len(parts[0])
        tokens2 = len(parts[1])
    else:
        tokenised = tokeniser(
           [parts[0]],
           is_split_into_words = True,
        )
        tokens1 = len(parts[0])
        tokens2 = 0
    length = len(tokenised['input_ids'][0])
    length2count[length] += 1             

    if length < 90:
        print(
            '\nFirst and last 3 sentences of doc_idx', doc_idx,
            'with', length, 'seq_len:'
        )
        sent_idx = set()
        for idx in range(3):
            sent_idx.add(idx)
            idx = len(raw_data[doc_idx][0]) - 1 - idx
            if idx >= 0:
                sent_idx.add(idx)
        for s_idx in sorted(list(sent_idx)):
            try:
                sentence = raw_data[doc_idx][0][s_idx]
                print('[%d] %r' %(s_idx, sentence))
            except IndexError:
                pass
        print('Selected slice 1:', parts[0])
        if len(parts) == 2:
            print('Selected slice 2:', parts[1])
    if doc_idx < 10:
        print('%7d %7d %7d %7d %5d' %(
            doc_idx, length, tokens1, tokens2, tokens1+tokens2,
        ))
        
print('\nFrequency of each sequence length:')        
for length in sorted(list(length2count.keys())):
    print(length, length2count[length])


# In[21]:


# https://github.com/ricardorei/lightning-text-classification/blob/master/classifier.py
    
import pytorch_lightning as pl
from torchnlp.encoders import LabelEncoder

class SlicedDataModule_Part_1(pl.LightningDataModule):
    
    def __init__(self, classifier, data_split = None, **kwargs):
        super().__init__()
        self.hparams = classifier.hparams
        self.classifier = classifier
        if data_split is None:
            # this happens when loading a checkpoint
            data_split = (None, None, None)
        elif len(data_split) == 2:
            # add empty validation set
            tr_data, val_data = self.split(data_split[0], 0.9)
            data_split = (tr_data, val_data, data_split[1])
        self.data_split = data_split
        self.kwargs = kwargs
        self.label_encoder = LabelEncoder(
            ['pos', 'neg'],
            reserved_labels = [],
        )

    def train_dataloader(self) -> DataLoader:
        assert self.hparams.batch_size <= batch_size
        dataset = SlicedDocuments(
            raw_data = self.data_split[0],
            **self.kwargs
        )
        return DataLoader(
            dataset = dataset,
            sampler     = RandomSampler(dataset),
            batch_size  = self.hparams.batch_size,
            collate_fn  = self.classifier.prepare_sample,
            num_workers = self.hparams.loader_workers,
        )
  


# In[22]:


class SlicedDataModule(SlicedDataModule_Part_1):
    
    def val_dataloader(self) -> DataLoader:
        assert self.hparams.batch_size <= batch_size
        if not self.data_split[1]:
            return None   # TODO: check documentation what to return
        return DataLoader(
            dataset = SlicedDocuments(
                raw_data = self.data_split[1],
                **self.kwargs
            ),
            batch_size  = self.hparams.batch_size,
            collate_fn  = self.classifier.prepare_sample,
            num_workers = self.hparams.loader_workers,
        )
    
    def test_dataloader(self) -> DataLoader:
        assert self.hparams.batch_size <= batch_size
        return DataLoader(
            dataset = SlicedDocuments(
                raw_data = self.data_split[2],
                **self.kwargs
            ),
            batch_size  = self.hparams.batch_size,
            collate_fn  = self.classifier.prepare_sample,
            num_workers = self.hparams.loader_workers,
        )
    
    def split(self, data, ratio):
        # get label distribution:
        distribution = defaultdict(lambda: 0)
        for _, label in data:
            distribution[label] += 1
        # get target frequencies of labels in first set
        still_needed = defaultdict(lambda: 0)
        for label in distribution:
            still_needed[label] = int(ratio*distribution[label])
        # split data accordingly
        dataset_1 = []
        dataset_2 = []
        for item in data:
            label = item[1]
            if still_needed[label] > 0:
                dataset_1.append(item)
                still_needed[label] -= 1
            else:
                dataset_2.append(item)
        return dataset_1, dataset_2


# In[23]:


from transformers import AutoModel
import torch.nn as nn

class Classifier_part_1(pl.LightningModule):
    
    def __init__(self, hparams = None, **kwargs) -> None:
        super().__init__()
        if type(hparams) is dict:
            #print('Converting', type(hparams))
            hparams = pl.utilities.AttributeDict(hparams)
        #print('New classifier with', hparams)
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.data = SlicedDataModule(self, **kwargs)
        if 'tokeniser' in kwargs:
            self.tokenizer = kwargs['tokeniser']  # attribute expected by lightning
        else:
            # this happens when loading a checkpoint
            self.tokenizer = None  # TODO: this may break ability to use the model
        self.__build_model()
        self.__build_loss()
        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs
        self.record_predictions = False
            
    def __build_model(self) -> None:
        ''' Init BERT model, tokeniser and classification head '''
        # Q: Why not use AutoModelForSequenceClassification?
        self.bert = AutoModel.from_pretrained(
            model_name,  # was: self.hparams.encoder_model
            output_hidden_states = True
        )
        self.classification_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.bert.config.hidden_size, 1536),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(1536, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, self.data.label_encoder.vocab_size)
        )
        
    def __build_loss(self):
        self._loss = nn.CrossEntropyLoss()


# In[24]:


from transformers import AutoModel
import torch.nn as nn

class Classifier_part_1(pl.LightningModule):
    
    def __init__(self, hparams = None, **kwargs) -> None:
        super().__init__()
        if type(hparams) is dict:
            #print('Converting', type(hparams))
            hparams = pl.utilities.AttributeDict(hparams)
        #print('New classifier with', hparams)
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.data = SlicedDataModule(self, **kwargs)
        if 'tokeniser' in kwargs:
            self.tokenizer = kwargs['tokeniser']  # attribute expected by lightning
        else:
            # this happens when loading a checkpoint
            self.tokenizer = None  # TODO: this may break ability to use the model
        self.__build_model()
        self.__build_loss()
        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs
        self.record_predictions = False
            
    def __build_model(self) -> None:
        ''' Init BERT model, tokeniser and classification head '''
        # Q: Why not use AutoModelForSequenceClassification?
        self.bert = AutoModel.from_pretrained(
            model_name,  # was: self.hparams.encoder_model
            output_hidden_states = True
        )
        self.classification_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.bert.config.hidden_size, 1536),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(1536, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, self.data.label_encoder.vocab_size)
        )
        
    def __build_loss(self):
        self._loss = nn.CrossEntropyLoss()


# In[25]:


import logging as log

class Classifier_part_2(Classifier_part_1):
    
    def unfreeze_encoder(self) -> None:
        if self._frozen:
            log.info('\n== Encoder model fine-tuning ==')
            for param in self.bert.parameters():
                param.requires_grad = True
            self._frozen = False
            
    def freeze_encoder(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, sample: dict) -> dict:
        if self.training:
            self.eval()
        with torch.no_grad():
            batch_inputs, _ = self.prepare_sample(
                [sample],
                prepare_target = False
            )
            model_out = self.forward(batch_inputs)
            logits = torch.Tensor.cpu(model_out["logits"]).numpy()
            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction]
                for prediction in numpy.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]
        return sample
    
    def start_recording_predictions(self):
        self.record_predictions = True
        self.reset_recorded_predictions()
        
    def stop_recording_predictions(self):
        self.record_predictions = False
        
    def reset_recorded_predictions(self):
        self.seq2label = {}


# In[26]:


from torchnlp.utils import lengths_to_mask

class Classifier_part_3(Classifier_part_2):
    
    def forward(self, batch_input):
        tokens  = batch_input['input_ids']
        lengths = batch_input['length']
        mask = batch_input['attention_mask']
        # Run BERT model.
        word_embeddings = self.bert(tokens, mask).last_hidden_state
        sentemb = word_embeddings[:,0]  # at position of [CLS]
        logits = self.classification_head(sentemb)
        # Hack to conveniently use the model and trainer to
        # get predictions for a test set:
        if self.record_predictions:
            logits_np = torch.Tensor.cpu(logits).numpy()
            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction]
                for prediction in numpy.argmax(logits_np, axis=1)
            ]
            for index, input_token_ids in enumerate(tokens):
                key = torch.Tensor.cpu(input_token_ids).numpy().tolist()
                # truncate trailing zeros
                while key and key[-1] == 0:
                    del key[-1]
                self.seq2label[tuple(key)] = predicted_labels[index]
        return {"logits": logits}
    
    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])


# In[27]:


class Classifier_part_4(Classifier_part_3):
    
    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        assert len(sample) <= batch_size
        assert self.tokenizer is not None
        with_1_part = 0
        with_2_parts = 0
        batch_part_1 = []
        batch_part_2 = []
        for item in sample:
            parts = item['parts']
            if len(parts) == 2:
                with_2_parts += 1
                batch_part_1.append(parts[0])
                batch_part_2.append(parts[1])
            else:
                with_1_part += 1
                batch_part_1.append(parts[0])
        assert not (with_1_part and with_2_parts)
        kwargs = {
            'is_split_into_words': True,
            'return_length':       True,
            'padding':             'max_length',
            # https://github.com/huggingface/transformers/issues/8691
            'return_tensors':      'pt',
        }
        if with_2_parts:
            encoded_batch = self.tokenizer(
                batch_part_1,
                batch_part_2,
                **kwargs
            )
        else:
            encoded_batch = self.tokenizer(
                batch_part_1,
                **kwargs
            )
        if not prepare_target:
            return encoded_batch, {}
        # Prepare target:
        batch_labels = []
        for item in sample:
            batch_labels.append(item['label'])
        assert len(batch_labels) <= batch_size
        try:
            targets = {
                "labels": self.data.label_encoder.batch_encode(batch_labels)
            }
            return encoded_batch, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")


# In[28]:


from collections import OrderedDict

class Classifier_part_5(Classifier_part_4):
    
    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # Q: What is this about?
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
        output = OrderedDict({"loss": loss_val})
        self.log('train_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True)
        # can also return just a scalar instead of a dict (return loss_val)
        return output
   
    def test_or_validation_step(self, test_type, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)
        y = targets["labels"]
        y_hat = model_out["logits"]
        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)
        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)
        output = OrderedDict({
            test_type + "_loss": loss_val,
            test_type + "_acc":  val_acc,
            'batch_size': len(batch),
            #'predictions': labels_hat,
        })
        return output
    
    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        return self.test_or_validation_step(
            'val', batch, batch_nb, *args, **kwargs
        )
    
    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        return self.test_or_validation_step(
            'test', batch, batch_nb, *args, **kwargs
        )


# In[29]:


from torch import optim

class Classifier(Classifier_part_5):
    
    # validation_end() is now validation_epoch_end()
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/efd272a3cac2c412dd4a7aa138feafb2c114326f/CHANGELOG.md
    
    def test_or_validation_epoch_end(self, test_type, outputs: list) -> None:
        val_loss_mean = 0.0
        val_acc_mean = 0.0
        total_size = 0
        for output in outputs:
            val_loss = output[test_type + "_loss"]
            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss
            # reduce manually when using dp
            val_acc = output[test_type + "_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)
            # We weight the batch accuracy by batch size to not give
            # higher weight to the items of a smaller, final bacth.
            batch_size = output['batch_size']
            val_acc_mean += val_acc * batch_size
            total_size += batch_size
        val_loss_mean /= len(outputs)
        val_acc_mean /= total_size
        self.log(test_type+'_loss', val_loss_mean)
        self.log(test_type+'_acc',  val_acc_mean)

    def validation_epoch_end(self, outputs: list) -> None:
        self.test_or_validation_epoch_end('val', outputs)
                                     
    def test_epoch_end(self, outputs: list) -> None:
        self.test_or_validation_epoch_end('test', outputs)
        
    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.bert.parameters(),
                "lr": self.hparams.encoder_learning_rate,
                #"weight_decay": 0.01,  # TODO: try this as it is in the BERT paper
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()


# In[30]:


xval_run = 0  # 0 to 9


#class DotDict(pl.utilities.AttributeDict):    
#    __getattr__ = dict.get

    # the above misses pickle support; presumable dict defines custom pickle behaviour
    # that causes the unpickled object to be a plain dict object

print('batch_size', batch_size)

classifier = Classifier(
    hparams = {
        "encoder_learning_rate": 1e-05,  # Encoder specific learning rate
        "learning_rate":         3e-05,  # Classification head learning rate
        "nr_frozen_epochs":      3,      # Number of epochs we want to keep the encoder model frozen
        "loader_workers":        4,      # How many subprocesses to use for data loading.
                                         # (0 means that the data will be loaded in the main process)
        "batch_size":            batch_size,
        "gpus":                  1,
    },
    # parameters for SlicedDataModule:
    data_split = splits[xval_run],
    # parameters for SlicedDocument():
    tokeniser                   = tokeniser,
    fraction_for_first_sequence = 0.0,   # set to 0.0001 to duplicate short documents
    max_sequence_length         = max_sequence_length,
    second_part_as_sequence_B   = False,
    preproc_batch_size          = 8
)   
print('Ready.')


# In[31]:


# https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

early_stop_callback = EarlyStopping(
    monitor   = 'val_acc',
    min_delta = 0.00,
    patience  = 5,
    verbose   = False,
    mode      = 'max',
)

save_top_model_callback = ModelCheckpoint(
    save_top_k = 3,
    monitor    = 'val_acc',
    mode       = 'max',
    filename   = '{val_acc:.4f}-{epoch:02d}-{val_loss:.4f}'
)

trainer = pl.Trainer(
    callbacks=[early_stop_callback, save_top_model_callback],
    max_epochs = 6,
    min_epochs = classifier.hparams.nr_frozen_epochs + 2,
    gpus = classifier.hparams.gpus,
    accumulate_grad_batches = 4,   # compensate for small batch size
    #limit_train_batches = 10,  # use only a subset of the data during development for higher speed
    check_val_every_n_epoch = 1,
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/6690
    logger = pl.loggers.TensorBoardLogger(os.path.abspath('lightning_logs')),
)
start = time.time()
trainer.fit(classifier, classifier.data)
print('Training time: %.0f minutes' %((time.time()-start)/60.0))


# In[32]:


import torch
torch.cuda.is_available()


# In[ ]:




