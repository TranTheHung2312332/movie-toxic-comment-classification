# Cách dùng

```
import pickle
import re
from itertools import chain
from vncorenlp import VnCoreNLP

rdrsegmenter = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg,pos,ner", max_heap_size='-Xmx2g')

with open("data/stopwords.txt", encoding='utf-8') as f:
    stopwords = {line.strip().lower() for line in f if line.strip()}

def preprocessing(sentence):
    # Loại bỏ emoji và một số kí tự khác
    s = re.sub(r'[^\w\s.,!?]|_', '', str(sentence))

    # Loại bỏ khoảng trắng đầu và cuối
    if not s.strip():
        return []

    s = s.strip()

    # NER và chuyển về chữ thường, loại bỏ stopwords
    tagged = rdrsegmenter.ner(s)
    tagged = list(chain.from_iterable(tagged))

    output = []
    for word, tag in tagged:
        if word.lower() in stopwords:
            continue

        if tag != 'O':
            output.append(f'<{tag}>')
        else:
            output.append(word.lower())
    
    return output

with open('embedding_models/bow.pkl', mode='rb') as f:
    bow = pickle.load(f)

vec = bow.transform(['Xin chào các bạn'])
```