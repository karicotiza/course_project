import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers, models, layers, backend, callbacks

print(
    f"TensorFlow - {tf.__version__}\n"
    f"Keras - {tf.keras.__version__}"
)

if tf.test.gpu_device_name():
    print("GPU - On")
    # TensorFlow / CUDA / CUDnn compatibility table - https://www.tensorflow.org/install/source#gpu
    print(f"CUDA - {tf.sysconfig.get_build_info()['cuda_version']}")
else:
    print("GPU - Off")

data = pd.read_csv("data/data.csv")

train_inputs, test_inputs, train_targets, test_targets = train_test_split(
    data["description"],
    data["industry"],
    test_size=0.2,
    random_state=42,
)
shape = train_inputs.shape[1:]

num_words = 25000
max_len = 60
nb_classes = 141

train_targets = to_categorical(np.asarray(train_targets.factorize()[0]), 141)
test_targets = to_categorical(np.asarray(test_targets.factorize()[0]), 141)

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data["description"])

sequences = tokenizer.texts_to_sequences(train_inputs)
train_inputs = pad_sequences(sequences, maxlen=max_len)

model = models.Sequential(name="industry_classification")

model.add(layers.Embedding(num_words, 32, input_length=max_len))
model.add(layers.GRU(512))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(units=nb_classes, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

checkpoint = callbacks.ModelCheckpoint(
    'model.hdf5',
    monitor='accuracy',
    verbose=True,
    save_best_only=True,
)

early_stop = callbacks.EarlyStopping(
    monitor='accuracy',
    patience=3,
    restore_best_weights=True
)

callbacks_list = [checkpoint, early_stop]

history = model.fit(
    train_inputs,
    train_targets,
    verbose=True,
    epochs=50,
    callbacks=[
        callbacks_list,
    ],
)
