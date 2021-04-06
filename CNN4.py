# Dima Oana-Teodora 241
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop

#import tensorflow.compat as tf
#import tensorflow.compat.v1 as tf 
#tf.disable_v2_behavior()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import csv

from sklearn.utils import shuffle

from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.layers.core import Dropout
from sklearn import metrics

# citire imagini de antrenare plus etichete de antrenare
f = open ("train.txt") # deschidere fisier
train_images = []
train_labels = []
for line in enumerate(open("train.txt", "r")):
    v = line[1].split(',') # citire linie din fisier
    image_name = 'train/'+v[0] # formare cale catre folderul cu imagini
    label = int(v[1]) # transformare numar clasa din string in int
    train_labels.append(label) # adaugare clasa 
    image = plt.imread(image_name) # citire imagine matrice 32 X 32
    
    image=image.reshape(-1) # din 2D => 1D
    
    train_images.append(image) # adaugare imagine
# transformare in numpy arrays
train_images = np.array(train_images) 
train_labels = np.array(train_labels) 

train_images, train_labels = shuffle(train_images, train_labels) # amestecare

# citire imagini de testare 
f = open ("test.txt")
test_images = []
name_test_images =[]
for line in enumerate(open("test.txt", "r")):
    image_name = 'test/'+ line[1][:len(line[1])-1] # formare cale director cu imagini 
                                                    #(+ sterg caracterul '\n' de la final, altfel => erooare)
    name_test_images.append(line[1][:len(line[1])-1])
    image = plt.imread(image_name)
    image=image.reshape(-1) # din 2D => 1D
    test_images.append(image)
    
test_images = np.array(test_images)

# citire imagini de validare plus etichete de validare
f = open ("validation.txt")
validation_images = []
validation_labels = []
for line in enumerate(open("validation.txt", "r")):
    v = line[1].split(',')
    image_name = 'validation/'+v[0]
    label = int(v[1])
    validation_labels.append(label)
    image = plt.imread(image_name)
    image=image.reshape(-1) # din 2D => 1D
    validation_images.append(image)

validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)

validation_images, validation_labels = shuffle(validation_images, validation_labels)

classes = 9 # 9 categorii notate de la 0 la 8

y_val=np_utils.to_categorical(validation_labels, classes) 
y_train=np_utils.to_categorical(train_labels, classes) 

# Marire esantion de date
# creez un esantion separat
# prin aplicarea de transformari aleatorii imaginilor de antrenament 
# Scop: evitarea overfitting-ului & expunerea modelului la mai multe cazuri de studiu  
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"), #rasucire orizontala
        layers.experimental.preprocessing.RandomRotation(0.1), #rotatie aleatorie a imaginii
    ]
)
# creare model de baza
base_model = keras.applications.Xception(
    weights="imagenet",  # incarcare date preantrenate din ImageNet
    input_shape=(150, 150, 3), #dimensiuni ImageNet
    include_top=False
)  


# "Dezactivare" model de baza (Freezing) 
base_model.trainable = False

# Craere model nou 
inputs = keras.Input(shape=(32, 32, 3)) 
x = data_augmentation(inputs)  # Aplicare transformari aleatorii 

# Adaug un layer de normalizare pe intervalele [0,255] & [-1,1] => datele preantrenate au nevoie de normalizare
norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5] * 3)
var = mean ** 2
# [-1,1]
x = norm_layer(x)
norm_layer.set_weights([mean, var])

# "Batch normalization" -> model mai rapid si stabil, normalizare layer de intrare (prin recentrare si redimensionare)
# Modelul de baza contine layere batch-norm 
# Pastrare model in modul de inferenta (inference mode) -> pregatirea pentru fine-tuning  
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)

# Pentru regularizare adaug un Dropout layer => Scop: reducere overfitting
x = keras.layers.Dropout(0.2)(x)  

outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary()

model = Sequential()
model.add(Dense(1000, input_shape =(train_images.shape[1],)))
model.add(Activation ('relu'))
model.add(Dense(200))
model.add(Activation ('sigmoid'))
model.add(Dense(classes))
model.add(Activation ('softmax'))

# configurare model
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-6, decay=1e-6), metrics=['accuracy'])
# antrenare model
history = model.fit(train_images, y_train, batch_size=200, epochs=100, verbose=2, validation_data=(validation_images, y_val))
# evaluare model
loss_accuracy = model.evaluate(validation_images, y_val)
print(loss_accuracy)

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(100)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(100)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# prezic clasele pe imaginile de validare
y_pred=model.predict_classes(validation_images).astype("int32")
# calcul matrice de confuzie
confusion_matrix=metrics.confusion_matrix(validation_labels, y_pred, labels=[0,1,2,3,4,5,6,7,8])
#print(confusion_matrix)

from mlxtend.plotting import plot_confusion_matrix

plot_confusion_matrix(confusion_matrix, colorbar=True, 
                      class_names=[0,1,2,3,4,5,6,7,8], 
                      figsize=(10, 10))
plt.show()

# prezicere
predict = model.predict_classes(test_images).astype("int32")

# scriere in fisier 
f = open("submission_2.csv", "w", newline="")
csv_file= csv.writer(f, delimiter=",") #stabilire delimitatori cuvinte
csv_file.writerow(['id', 'label']) # capul de tabel
for i in range (predict.shape[0]):
    csv_file.writerow([name_test_images[i], predict[i]])
f.close()