import tokenizers

MAX_LEN = 192
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 5
ROBERTA_PATH = "/content/drive/My Drive/pre_trained_model/roberta-base"
TRAINING_FILE = "../input/train_folds.csv"
TEST_FILE = "../input/test.csv"
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{ROBERTA_PATH}/vocab.json", 
    merges_file=f"{ROBERTA_PATH}/merges.txt", 
    lowercase=True,
    add_prefix_space=True
)