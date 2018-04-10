from hmlstm.data import CBTDataset

CBTDataset.create_vocabulary('./CBTest/data/cbtest_CN_train.txt', './CBT_CN_vocab.txt')
CBTDataset.create_vocabulary('./CBTest/data/cbtest_NE_train.txt', './CBT_NE_vocab.txt')