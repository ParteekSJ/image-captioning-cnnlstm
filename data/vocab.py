from pycocotools.coco import COCO
from collections import Counter
import nltk
import os, pickle


class Vocab(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.index = 0

    def __call__(self, token):
        if not token in self.w2i:
            return self.w2i["<unk>"]
        return self.w2i[token]

    def __len__(self):
        return len(self.w2i)

    def add_token(self, token):
        if not token in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index += 1


def build_vocabulary(json_paths, threshold, logger):
    """Build a simple vocabulary wrapper."""
    # Initialize a Counter to count word occurrences
    word_counter = Counter()

    # Loop through the list of JSON paths
    for json_path in json_paths:
        coco = COCO(json_path)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]["caption"])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if (i + 1) % 1000 == 0:
                logger.debug(
                    ("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))
                )

        # Merge the word counts into the main word_counter
        word_counter.update(counter)

    # If the word frequency is less than 'threshold', then the word is discarded.
    tokens = [token for token, cnt in word_counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocab()
    vocab.add_token("<pad>")
    vocab.add_token("<start>")
    vocab.add_token("<end>")
    vocab.add_token("<unk>")

    # Add the words to the vocabulary.
    for i, token in enumerate(tokens):
        vocab.add_token(token)
    return vocab


def save_and_retrieve_vocab(cfg, logger):
    vocab_save_path = cfg.DATASET.SAVED_VOCAB_PATH
    json_vocab_paths = [cfg.DATASET.ANNOTATIONS_DIR, cfg.DATASET.VAL_ANNOTATIONS_DIR]
    threshold = cfg.DATASET.THRESHOLD

    # Check if the Pickle file exists
    if os.path.exists(vocab_save_path):
        # If the file exists, load and return the vocabulary
        with open(vocab_save_path, "rb") as f:
            vocab = pickle.load(f)
        logger.info(f"Loaded the vocabulary from: {vocab_save_path}")
    else:
        # If the file does not exist, perform vocabulary computation
        vocab = build_vocabulary(
            json_paths=json_vocab_paths, threshold=threshold, logger=logger
        )
        # Save the computed vocabulary to the Pickle file
        with open(vocab_save_path, "wb") as f:
            pickle.dump(vocab, f)
        logger.info(f"Total vocabulary size: {len(vocab)}")
        logger.info(f"Saved the vocabulary wrapper to: {vocab_save_path}")

    return vocab
