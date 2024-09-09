import os
import sys
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class TrainWord2Vec(object):
    "class to train word2vec vectorization"
    def __init__(self, path=None, sep=None):
        """Options to choose the separator and the path (both must be given as input in the terminal)
        """
        sys.stdout.write('Reading input file...\n'); sys.stdout.flush()

        # read the training df
        self.__df = pd.read_csv(path, sep=sep)
      
    def tokenize_text(self):
        self.__df['w2v_tokenized_text'] = self.__df['text'].astype(str).apply(gensim.utils.simple_preprocess)
        return 

    def train_word2vec(self):
        # Model parameters
        # window: number of words next to the target word to consider - kept constant as 3 right now
        # min_count: minimum count of words to consider when training the model (word appears in at least X docs) - also kept constant for comparison
        # sg: algorithm, 0 for CBOW and 1 for skipgram - always using CBOW
        sys.stdout.write('Training Word2Vec on the corpus...\n'); sys.stdout.flush()
        model_w2v = gensim.models.Word2Vec(window=3, min_count=10, workers=1, sg=0) 
        model_w2v.build_vocab(self.__df.w2v_tokenized_text, progress_per=1000)
        model_w2v.train(self.__df.w2v_tokenized_text, total_examples=model_w2v.corpus_count, epochs=model_w2v.epochs) # I think default epochs is 5
        sys.stdout.write('Computing document vectors...\n'); sys.stdout.flush()
        self.model_w2v = model_w2v
        return 

    def store_w2v(self, path):
        sys.stdout.write('Saving dataframe with the tokenized text as csv...\n'); sys.stdout.flush()
        self.__df.to_csv(path, index=False)
        sys.stdout.write('Dataframe saved.\n'); sys.stdout.flush()
        
        sys.stdout.write('Saving trained model...'); sys.stdout.flush()
        self.model_w2v.save("../models/word2vec_model.model")
        sys.stdout.write('Model saved.\n'); sys.stdout.flush()

        return