
# coding: utf-8

# In[ ]:


import os
import csv
import sys
import multiprocessing
sys.path


# In[ ]:


size_of_image=input("What do you want the resized images to resize to: ")
if "none" in size_of_image.lower():
	size_of_image=None 
else:
	size_of_image = int(size_of_image)

# In[ ]:


hours_for_training=float(input("How many hours to train (includes decimals)?"))


# In[ ]:

print("Splitting Folders")
import split_folders
split_folders.ratio('data', output="split_data", ratio=(.8, .1, .1)) 


# In[ ]:

print("Renaming Files to Convention")
get_ipython().system(u"cd split_data/train && find . -type f -name '*.JPG' -print0 | xargs -0 rename 's/\\.JPG/\\.jpeg/'")
get_ipython().system(u"cd split_data/train && find . -type f -name '*.jpg' -print0 | xargs -0 rename 's/\\.jpg/\\.jpeg/'")
get_ipython().system(u"cd split_data && find train -name '*.jpeg' -exec changeName.sh {} \;")
get_ipython().system(u'cd split_data/train && for d in ./*/ ; do (cd "$d" && find . -type f ! -name "*.jpeg" -exec rm {} \;); done')


get_ipython().system(u"cd split_data/val && find . -type f -name '*.JPG' -print0 | xargs -0 rename 's/\\.JPG/\\.jpeg/'")
get_ipython().system(u"cd split_data/val && find . -type f -name '*.jpg' -print0 | xargs -0 rename 's/\\.jpg/\\.jpeg/'")
get_ipython().system(u"cd split_data && find val -name '*.jpeg' -exec changeName.sh {} \;")
get_ipython().system(u'cd split_data/val && for d in ./*/ ; do (cd "$d" && find . -type f ! -name "*.jpeg" -exec rm {} \;); done')


get_ipython().system(u"cd split_data/test && find . -type f -name '*.JPG' -print0 | xargs -0 rename 's/\\.JPG/\\.jpeg/'")
get_ipython().system(u"cd split_data/test && find . -type f -name '*.jpg' -print0 | xargs -0 rename 's/\\.jpg/\\.jpeg/'")
get_ipython().system(u"cd split_data && find test -name '*.jpeg' -exec changeName.sh {} \;")
get_ipython().system(u'cd split_data/test && for d in ./*/ ; do (cd "$d" && find . -type f ! -name "*.jpeg" -exec rm {} \;); done')

# In[ ]:


train_dir = 'split_data/train' # Path to the train directory
class_dirs = [i for i in os.listdir(path=train_dir) if os.path.isdir(os.path.join(train_dir, i))]
with open('split_data/train/label.csv', 'w') as train_csv:
    fieldnames = ['File Name', 'Label']
    writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
    writer.writeheader()
    label = 0
    for current_class in class_dirs:
        for image in os.listdir(os.path.join(train_dir, current_class)):
            writer.writerow({'File Name': str(image), 'Label':label})
        label += 1
    train_csv.close()


# In[ ]:

print("Resizing Files")
from PIL import Image
import os, sys
import time
from tqdm import tqdm
get_ipython().system(u'mkdir split_data/train_all')
get_ipython().system(u'find split_data/train/ -type f -print0 | xargs -0 cp -t split_data/train_all')
get_ipython().system(u'cd split_data/train_all && rm *.csv')
path = "split_data/train_all/"
dirs = os.listdir( path )

def resize(item):
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((size_of_image,size_of_image), Image.ANTIALIAS)
            try:
                imResize.save(f+"smaller.jpeg", 'JPEG', quality=100)
            except:
                imResize = imResize.convert("RGB")
                imResize.save(f+"smaller.jpeg", 'JPEG', quality=100)


get_ipython().system(u'mkdir split_data/resized-train/')

if size_of_image == None:	
	get_ipython().system(u'mv split_data/train_all/*.jpeg split_data/resized-train/')
	
else:
	processes = []
	pool = multiprocessing.Pool()
	print(dirs)
	pool.map(resize, dirs)
	pool.close()
	get_ipython().system(u'find split_data/train_all -name "*smaller.jpeg" -exec mv {} split_data/resized-train \;')
	get_ipython().system(u'cd split_data/resized-train && for f in *smaller*; do     mv -- "$f" "${f/smaller/}"; done')


# In[ ]:


train_dir = 'split_data/val' # Path to the train directory
class_dirs = [i for i in os.listdir(path=train_dir) if os.path.isdir(os.path.join(train_dir, i))]
with open('split_data/val/label.csv', 'w') as train_csv:
    fieldnames = ['File Name', 'Label']
    writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
    writer.writeheader()
    label = 0
    for current_class in class_dirs:
        for image in os.listdir(os.path.join(train_dir, current_class)):
            writer.writerow({'File Name': str(image), 'Label':label})
        label += 1
    train_csv.close()


# In[ ]:
get_ipython().system(u'mkdir split_data/val_all')
get_ipython().system(u'find split_data/val/ -type f -print0 | xargs -0 cp -t split_data/val_all')
get_ipython().system(u'cd split_data/val_all && rm *.csv')
get_ipython().system(u'mkdir split_data/resized-val')

if size_of_image == None:
	get_ipython().system(u'mv split_data/val_all/*.jpeg split_data/resized-val/')
	
else:
	path = "split_data/val_all/"
	dirs = os.listdir( path )

	processes = []
	pool = multiprocessing.Pool()
	pool.map(resize, dirs)
	pool.close()

	get_ipython().system(u'find split_data/val_all -name "*smaller.jpeg" -exec mv {} split_data/resized-val \;')
	get_ipython().system(u'cd split_data/resized-val && for f in *smaller*; do     mv -- "$f" "${f/smaller/}"; done')



# In[ ]:


train_dir = 'split_data/test' # Path to the train directory
class_dirs = [i for i in os.listdir(path=train_dir) if os.path.isdir(os.path.join(train_dir, i))]
with open('split_data/test/label.csv', 'w') as train_csv:
    fieldnames = ['File Name', 'Label']
    writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
    writer.writeheader()
    label = 0
    for current_class in class_dirs:
        for image in os.listdir(os.path.join(train_dir, current_class)):
            writer.writerow({'File Name': str(image), 'Label':label})
        label += 1
    train_csv.close()


# In[ ]:


get_ipython().system(u'mkdir split_data/test_all')
get_ipython().system(u'find split_data/test/ -type f -print0 | xargs -0 cp -t split_data/test_all')
get_ipython().system(u'cd split_data/test_all && rm *.csv')
get_ipython().system(u'mkdir split_data/resized-test')

if size_of_image == None:
	get_ipython().system(u'mv split_data/test_all/*.jpeg split_data/resized-test/')
	
else:
	path = "split_data/test_all/"
	dirs = os.listdir( path )

	processes = []
	pool = multiprocessing.Pool()
	pool.map(resize, dirs)
	pool.close()
	
	get_ipython().system(u'find split_data/test_all -name "*smaller.jpeg" -exec mv {} split_data/resized-test \;')
	get_ipython().system(u'cd split_data/resized-test && for f in *smaller*; do     mv -- "$f" "${f/smaller/}"; done')


# In[ ]:

print("Loading data into memory")
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

