
# coding: utf-8

# In[ ]:


import os
import csv
import sys
import multiprocessing
sys.path


# In[ ]:


size_of_image=1024


# In[ ]:


hours_for_training=float(input("How many hours to train (in decimals)?"))



# In[ ]:


from autokeras.image.image_supervised import load_image_dataset


# In[ ]:


x_train, y_train = load_image_dataset(csv_file_path="split_data/train/label.csv",
                                      images_path="split_data/resized-train/")
print(x_train.shape)
print(y_train.shape)


# In[ ]:


from autokeras.image.image_supervised import ImageClassifier


# In[ ]:


get_ipython().system(u'mkdir models')
clf = ImageClassifier(path="models/",verbose=True)
clf.fit(x_train, y_train, time_limit=hours_for_training*60*60)


# In[ ]:


x_val, y_val = load_image_dataset(csv_file_path="split_data/val/label.csv",
                                      images_path="split_data/resized-val/")
print(x_val.shape)
print(y_val.shape)


# In[ ]:


clf.final_fit(x_train, y_train, x_val, y_val, retrain=True)
y = clf.evaluate(x_val, y_val)
print(y)


# In[ ]:


x_test, y_test = load_image_dataset(csv_file_path="split_data/test/label.csv",
                                      images_path="split_data/resized-test/")
print(x_test.shape)
print(y_test.shape)


# In[ ]:


y = clf.evaluate(x_test, y_test)
print(y)


# In[ ]:


import os
from graphviz import Digraph

from autokeras.utils import pickle_from_file


def to_pdf(graph, path):
    dot = Digraph(comment='The Round Table')

    for index, node in enumerate(graph.node_list):
        dot.node(str(index), str(node.shape))

    for u in range(graph.n_nodes):
        for v, layer_id in graph.adj_list[u]:
            dot.edge(str(u), str(v), str(graph.layer_list[layer_id]))

    dot.render(path)


def visualize(path):
    cnn_module = pickle_from_file(os.path.join(path, 'module'))
    cnn_module.searcher.path = path
    for item in cnn_module.searcher.history:
        model_id = item['model_id']
        graph = cnn_module.searcher.load_model_by_id(model_id)
    to_pdf(graph, os.path.join(path, str(model_id)))


# In[ ]:


visualize('models/')

