import os
import pathlib
import numpy as np
import shutil
import tensorflow as tf
from keras.models import load_model

from my_app.models import upload_img

img_height = 180
img_width = 180
batch_size = 32 

def validate():
    try:
        dataset_path = '../../../Documents/Malaria Dataset/validation dataset/' # ChangeMe
        data_dir = pathlib.Path(dataset_path).with_suffix('')

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        
        class_names = train_ds.class_names

        val_model_path = 'ML_Pred_Malaria_Backend/trained_model/val_model.keras'
        model = load_model(val_model_path)

        #model.evaluate(train_ds)
        model.summary() 

        #input_dir = 'input/'
        input_dir = 'media/uploaded_img/' # User input images

        print('VALIDATE IMAGE')
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(input_dir, filename)
                img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                confidence = ("{:.2f}".format(100 * np.max(score)))

                print("Image: ", filename)
                print(f'{class_names[np.argmax(score)]} = {score}')
                print(f"Confidence Level: {confidence}")
                print("------------------------------------")

                predicted_class = class_names[np.argmax(score)]

                if predicted_class == 'invalid':
                    # directory
                    old_path = f'media/uploaded_img/{filename}'
                    new_path = f'media/validation/invalid/{filename}'
                    os.rename(old_path, new_path)
                    # model
                    try:
                        update_old_path = f'uploaded_img/{filename}'
                        update_new_path = f'validation/invalid/{filename}'
                        update = upload_img.objects.get(image=update_old_path)
                        update.image = update_new_path
                        update.status = 'invalid'
                        update.con_lvl = confidence
                        update.save()
                    except upload_img.DoesNotExist:
                        print('No record found')

                elif predicted_class == 'valid':
                    # directory
                    old_path = f'media/uploaded_img/{filename}'
                    new_path = f'media/validation/valid/{filename}'
                    os.rename(old_path, new_path)
                    # model
                    try:
                        update_old_path = f'uploaded_img/{filename}'
                        update_new_path = f'validation/valid/{filename}'
                        update = upload_img.objects.get(image=update_old_path)
                        update.image = update_new_path
                        update.status = 'valid'
                        update.con_lvl = confidence
                        update.save()
                    except upload_img.DoesNotExist:
                        print('No Record Found')

                else:
                    print('Error!')

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
        dataset_path = '../../../Documents/Malaria Dataset/Dataset Pixel/' # ChangeMe
        data_dir = pathlib.Path(dataset_path).with_suffix('')

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        class_names = train_ds.class_names

        val_model_path = "ML_Pred_Malaria_Backend/trained_model/pred_model.keras"
        model = load_model(val_model_path)

        #model.evaluate(train_ds)
        #model.summary() 

        #input_dir = "src/valid_input/"
        input_dir = 'media/validation/valid/'

        print('THIN BLOOD SMEAR PREDICTION')
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(input_dir, filename)
                img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                confidence = ("{:.2f}".format(100 * np.max(score)))

                print("Image: ", filename)
                print(f'Prediction: {class_names[np.argmax(score)]}')
                print("Confidence Level: {:.2f}%".format(100 * np.max(score)))
                print("------------------------------------")

                predicted_class = class_names[np.argmax(score)]

                if predicted_class == 'Uninfected':
                    # directory
                    old_path = f'media/validation/valid/{filename}'
                    new_path = f'media/uninfected/{filename}'
                    os.rename(old_path, new_path)
                    # model
                    try:
                        update_old_path = f'validation/valid/{filename}'
                        update_new_path = f'uninfected/{filename}'
                        update = upload_img.objects.get(image=update_old_path)
                        update.image = update_new_path
                        update.label = 'uninfected'
                        update.con_lvl = confidence
                        update.save()
                    except upload_img.DoesNotExist:
                        print('No Record Found')

                elif predicted_class == 'Parasitized':
                    # directory
                    old_path = f'media/validation/valid/{filename}'
                    new_path = f'media/parasitized/{filename}'
                    os.rename(old_path, new_path)
                    # model
                    try:
                        update_old_path = f'validation/valid/{filename}'
                        update_new_path = f'parasitized/{filename}'
                        update = upload_img.objects.get(image=update_old_path)
                        update.image = update_new_path
                        update.label = 'parasitized'
                        update.con_lvl = confidence
                        update.save()
                    except upload_img.DoesNotExist:
                        print('No Record Found')
                else:
                    print('Error!')

    except Exception as e:
        print(e)

""" if __name__ == '__main__':
    validate()
    predict() """
