# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train,test, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--mode', type=str, default='train', required=True, help='train or test')
parser.add_argument('--emb_name', type=str, default='embedding', help='embedding layer name')
parser.add_argument('--attack', type=str, help='pgd or fgsm or free')
parser.add_argument('--attack_iter', type=int, default=3, help='pgd or free attack num')
parser.add_argument('--epsilon', type=float, default=0.8, help='epsilon')
parser.add_argument('--alpha', type=float, default=0.2, help='pgd alpha')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding, args.attack)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    if args.mode == 'train':
        train(config, model, train_iter, dev_iter, test_iter,emb_name=args.emb_name, attack=args.attack, attack_iter = args.attack_iter, epsilon=args.epsilon, alpha=args.alpha)
    else:
        test(config, model, test_iter)
