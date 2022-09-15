# %% [markdown]
import tensorflow as tf
import numpy as np
"""
Homework:

The folder '~//data//homework' contains a folder 'Data', containing hand-digits of letters a-z stored in .txt.

Try to establish a network to classify the digits.

`dataLoader.py` offers APIs for loading data.
"""
# %%
import dataLoader as dl
features,labels=dl.readData('../data/homework/Data')

class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# %%
import matplotlib.pyplot as plt
plt.plot(features[5,0:30],features[5,30:])
plt.suptitle="Real: "+labels[5]
plt.show()

# %%
# 训练集与测试集分离
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=1)

# %%
# 将 labels 中的字母转换为索引值
labels=[]
for letter in labels_train:
    labels.append(dl.letter2Number(letter))
labels_train1=np.array(labels)

labels=[]
for letter in labels_test:
    labels.append(dl.letter2Number(letter))
labels_test1=np.array(labels)# %%
# build the network
# 进行神经网络搭建
model = tf.keras.Sequential([
    tf.keras.Input(shape=(60)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(26, activation=tf.nn.softmax)
])
# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
# fit模型, 开始训练
model.fit(features_train, labels_train1, batch_size=3 epochs=10)
# training
# %%
# 测试模型
test_loss, test_acc = model.evaluate(features_test,  labels_test1)
print('Test accuracy:', test_acc)
labels_letter=model.predict(features_test)
labels_letter=np.argmax(labels_letter,axis=1)
acc=sum((labels_letter==labels_test1).tolist())/labels_letter.size
print('Test accuracy:', acc)
labels_hat_letter=[]
for num in labels_letter:
    labels_hat_letter.append(dl.number2Letter(num))
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
for i in range(num_images):
    plt.plot(features_test[i,0:30],features_test[i,30:])
    result="Real"+labels_test[i]+"Predicted:"+labels_hat_letter[i]
    print(result)
    plt.suptitle=result
    plt.show()
# %%
from sklearn.metrics import confusion_matrix
con_mat=confusion_matrix(labels_test,labels_hat_letter,labels=dl.getAlphabet(),normalize="true")
plt.matshow(con_mat)
plt.xticks(np.arange(26),dl.getAlphabet())
plt.yticks(np.arange(26),dl.getAlphabet())
plt.show()
# predict and evaluate



