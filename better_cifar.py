from __future__ import print_function
from layers.full import FullLayer
from layers.softmax import SoftMaxLayer
from layers.cross_entropy import CrossEntropyLayer
from layers.sequential import Sequential
from layers.relu import ReluLayer
from layers.dataset import cifar100
from sklearn.metrics import accuracy_score
import numpy as np

( x_train , y_train ) , ( x_test , y_test ) = cifar100(1213391684)

layer1 = FullLayer(3072,2052)
relu1 = ReluLayer()
layer2 = FullLayer(2052,680)
relu2 = ReluLayer()
layer3 = FullLayer(680,4)
softmax = SoftMaxLayer()
loss = CrossEntropyLayer()
model = Sequential((layer1,relu1,layer2,relu2,layer3,softmax),loss)

model.fit(x_train,y_train,epochs=25,lr=0.06)
out=model.predict(x_test)
y=np.reshape(y_test,(x_test.shape[0],1))
print("y_test.shape:",y.shape)
print("out.shape:",out.shape)
t= accuracy_score(y,out)
print(y_test , "Predicted Val",out)
print('accuracy score',t)
score=(np.mean(out==y))
print("score",score)
