# Image-to-Image-Translation-with-cGAN 
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the facades dataset (example of image-to-image translation)
dataset, info = tfds.load('facades', with_info=True, as_supervised=True)

train = dataset['train'].map(lambda inp, tar: (tf.image.resize(inp, [256, 256]) / 127.5 - 1,
                                               tf.image.resize(tar, [256, 256]) / 127.5 - 1))

# Load pix2pix model
OUTPUT_CHANNELS = 3
generator = tf.keras.models.load_model('https://storage.googleapis.com/tensorflow-models/savedmodel/pix2pix/pix2pix_generator')

# Pick one sample from dataset
for input_image, target_image in train.take(1):
    prediction = generator(tf.expand_dims(input_image, 0), training=True)

    plt.figure(figsize=(12, 4))

    display_list = [input_image, target_image, tf.squeeze(prediction)]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow((display_list[i] + 1) / 2.0)
        plt.axis('off')

    plt.show()
    
