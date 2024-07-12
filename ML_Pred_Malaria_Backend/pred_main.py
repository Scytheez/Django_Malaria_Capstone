import os
import pathlib
import numpy as np
import shutil
import tensorflow as tf
from keras.models import load_model

img_height = 180
img_width = 180
batch_size = 32 

def validate():
    try:
        dataset_path = '../../../Documents/Malaria Dataset/validation dataset'
        data_dir = pathlib.Path(dataset_path).with_suffix('')

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        
        class_names = train_ds.class_names

        val_model_path = "trained_model/val_model.keras"
        model = load_model(val_model_path)

        #model.evaluate(train_ds)
        model.summary() 

        input_dir = "input/" # User input images

        print('VALIDATE IMAGE')
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(input_dir, filename)
                img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                print("Image: ", filename)
                print(f'{class_names[np.argmax(score)]} = {score}')
                print("Confidence Level: {:.2f}%".format(100 * np.max(score)))
                print("------------------------------------")

                if class_names[np.argmax(score)] == 'invalid':
                    shutil.move(img_path, 'src/invalid_input/')
                elif class_names[np.argmax(score)] == 'valid':
                    shutil.move(img_path, 'src/valid_input/')
                else:
                    print('Unknown!')

    except Exception as e:
        print(e)

    finally:
        print()
        print()
        print('===============================================================')
        print()
        print()

def predict():
    try:
        dataset_path = '../../../Documents/Malaria Dataset/Dataset Pixel'
        data_dir = pathlib.Path(dataset_path).with_suffix('')

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        class_names = train_ds.class_names

        val_model_path = "trained_model/pred_model.keras"
        model = load_model(val_model_path)

        #model.evaluate(train_ds)
        #model.summary() 

        input_dir = "src/valid_input/"

        print('THIN BLOOD SMEAR PREDICTION')
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(input_dir, filename)
                img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                print("Image: ", filename)
                print(f'Prediction: {class_names[np.argmax(score)]}')
                print("Confidence Level: {:.2f}%".format(100 * np.max(score)))
                print("------------------------------------")

                """ 
                if class_names[np.argmax(score)] == 'Uninfected':
                    # do something
                elif class_names[np.argmax(score)] == 'Parasitized':
                    # do somthing
                else:
                    print('Unknown!') 
                """
    except Exception as e:
        print(e)

if __name__ == '__main__':
    validate()
    predict()