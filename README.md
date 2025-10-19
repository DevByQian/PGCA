This is our implementation for the paper:

_PGCA: A Price-Guided Contrastive Attention Network for Session-Based Recommendation

Jiansheng Qian, Huanhuan Yuan, Yongjing Hao, Guanfeng Liu, Pengpeng Zhao

Environments: Python3.12, Pytorch 2.3.0, CUDA 12.1

Download path of datasets:

  Electronics (events.csv): https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store

  Multi-category (2019-Oct.csv): https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

  Cosmetics (2019-Oct.csv): https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop

We also uploaded all three preprocessed datasets under the folder 'datasets'. You can directly use them to reproduce our results. You can also process your own datasets via the preprocess code we provide in the file 'datasets'.

Train the model:
~~~~
python main.py --dataset electronics --layer 1 --num_heads 4 --lambda1 0.1 --lambda2 0.1
python main.py --dataset multi-category --layer 2 --num_heads 4 --lambda1 0.05 --lambda2 0.8
python main.py --dataset cosmetics --layer 3 --num_heads 4 --lambda1 0.08 --lambda2 0.5   
~~~~