# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
print('ЗАГРУЗКА ДАННЫХ')
# Load data from the local folder 'flowers'
train_ds, eval_ds = tf.keras.utils.image_dataset_from_directory(
    directory = 'flowers',
    labels='inferred',
    label_mode='int',
    class_names=class_names,
    color_mode='rgb',
    image_size=(256, 256),
    shuffle=True,
    seed=321,
    validation_split=0.25,
    subset='both',
    crop_to_aspect_ratio=True)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i].numpy().astype("uint8"), cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()

answer = input("Загрузить ранее сохраненную модель с диска (y/n)? ")
if (answer == "y" or answer == "Y" or answer == ""):
    model = tf.keras.models.load_model("2nd_model")
else:
    model = tf.keras.Sequential([
    	tf.keras.layers.Rescaling(1./255),
    	tf.keras.layers.Conv2D(32, 3, activation='relu'),
    	tf.keras.layers.MaxPooling2D(),
    	tf.keras.layers.Conv2D(64, 3, activation='relu'),
    	tf.keras.layers.MaxPooling2D(),
    	tf.keras.layers.Conv2D(128, 3, activation='relu'),
    	tf.keras.layers.MaxPooling2D(),
    	tf.keras.layers.Flatten(input_shape=(256, 256, 3)), # 3 for rgb images
    	tf.keras.layers.Dense(256, activation='relu'),
    	tf.keras.layers.Dense(len(class_names))
	])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(train_ds, epochs=6)
    model.save("2nd_model")


print('\n')
eval_loss, eval_acc = model.evaluate(eval_ds, None, verbose=2)

print("\nTest loss: ", eval_loss, "Test accuracy: ", eval_acc)

print(model.summary())

# Predictions
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

my_flowers_ds = tf.keras.utils.image_dataset_from_directory(
    directory = 'my_flowers',
    labels='inferred',
    label_mode='int',
    class_names=class_names,
    color_mode='rgb',
    shuffle=False,
    image_size=(256, 256),
    crop_to_aspect_ratio=True)

#my_flowers_ds = eval_ds

predictions = probability_model.predict(my_flowers_ds)

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img.numpy().astype("uint8"), cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(5))
  plt.yticks([])
  thisplot = plt.bar(range(5), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 2
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for my_images, my_labels in my_flowers_ds.take(1):
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], my_labels, my_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], my_labels)
    plt.tight_layout()
    plt.show()
