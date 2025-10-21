# Cách dùng

## 1. Bag of Words và TF-IDF

```
from spicy import sparse

x_train = sparse.load_npz('embedded/tfidf/train.npz')
x_valid = sparse.load_npz('embedded/tfidf/valid.npz')
x_test = sparse.load_npz('embedded/tfidf/test.npz')
```

## 2. multilingual-e5-large

```
import numpy as np

x_train = np.load('embedded/multilingual-e5-large/train.npy')
x_valid = np.load('embedded/multilingual-e5-large/valid.npy')
x_test = np.load('embedded/multilingual-e5-large/test.npy')
```