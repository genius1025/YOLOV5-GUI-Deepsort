#%%
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# %%
#下载数据集
from keras.datasets import cifar10
# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
# summarize the dataset
print('Datatype: X:%s, y:%s' % (trainX.dtype,trainy.dtype))
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

# %%
# 抽查审视数据
from matplotlib import pyplot
for i in range(9):
	# define subplot
	pyplot.subplot(330+1+i) 
	# plot raw pixel data
	pyplot.imshow(trainX[i])
# show the figure
pyplot.show()
print(trainy[:8])

# %%
#预处理

from keras.utils import to_categorical
trainy = to_categorical(trainy)
testy = to_categorical(testy)
trainX = trainX / 255.0
testX = testX / 255.0
print(trainy[:8])

# %%
#写卷积神经网络模型
model = models.Sequential()
# 32 kernels of size (3,3)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

# %%
#训练模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(trainX, trainy, epochs=10, batch_size=64,validation_data=(trainX, trainy))
# %%
#评估模型
loss, acc = model.evaluate(testX, testy, verbose=0)
print('the accuracy of test set is: %.3f' % (acc * 100.0))

# 画图
import sys # 涉及系统中文件
# plot loss
pyplot.subplot(211)
pyplot.title('Cross Entropy Loss')
pyplot.plot(history.history['loss'], color='blue', label='train')
pyplot.plot(history.history['val_loss'], color='orange', label='test')
# plot accuracy
pyplot.subplot(212)
pyplot.title('Classification Accuracy')
pyplot.plot(history.history['accuracy'], color='blue', label='train')
pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
# pyplot.legend
pyplot.show
# save plot to file
#filename = sys.argv[0].split('/')[-1]
#pyplot.savefig(./ + '_plot.png')
#pyplot.close()
# %%
