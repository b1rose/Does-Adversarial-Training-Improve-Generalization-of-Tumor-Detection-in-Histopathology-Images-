import tensorflow as tf
import numpy as np
import tensorflow.keras
import glob
import os
import random
import pandas as pd
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgba2rgb
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.utils import shuffle
from scipy.stats import wasserstein_distance

import tensorflow as tf
print(tf.keras.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def create_DenseNet121(input_shape =(96,96,3)):


    i = Input(shape=input_shape)
    x = tf.cast(i, tf.float32)
    x = tf.keras.applications.densenet.preprocess_input(x)
    base_model = DenseNet121(weights=None,     
    include_top=False, input_shape=input_shape)
    x = base_model(x)
    x= GlobalAveragePooling2D()(x)
    preds=Dense(2,activation='softmax')(x)
    model = tf.keras.Model(inputs=[i], outputs=[preds])
    model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def generate_image_adversary(model, image, label, eps = 8, loss_object = tf.keras.losses.CategoricalCrossentropy()):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        with tf.device("/device:GPU:0"):
            tape.watch(image)
            pred = model(image)
            loss = loss_object(label, pred)
            gradient = tape.gradient(loss, image)
            signedGrad = tf.sign(gradient)
    adversary = np.clip((image + (signedGrad * eps)).numpy(), 0, 255).astype("uint8")
    return adversary

def generate_noisy_image(image, eps= 8):
    image = tf.cast(image, tf.float32)
    random_noise = np.random.randint(0, 2, (1, 96, 96, 3))
    random_noise[random_noise == 0] = -1
    noisy_image = np.clip((image + (random_noise * eps)).numpy(), 0, 255).astype("uint8")
    return noisy_image

def generate_data(batchSize, data_frame):

    amountOfImages = len(data_frame)

    while(1):
        batchStartIndex = 0
        batchEndIndex = batchSize

        while (batchStartIndex < amountOfImages):
            endIndex = 0
            if(batchEndIndex < amountOfImages):
                endIndex = batchEndIndex
            else:
                endIndex = amountOfImages

            #Load in batch of images and store them in imagesData
            imagesData = []
            targetsData = []
            for i in range(batchStartIndex,endIndex):    
                data_frame.iloc[i]
                patient_number = "{:03}".format(data_frame.iloc[i, 1])
                node_number = data_frame.iloc[i, 2]
                x_coord = data_frame.iloc[i, 3]
                y_coord = data_frame.iloc[i, 4]
                targetsData.append(data_frame.iloc[i, 5])
                image_folder_name = "patient_" + str(patient_number) + "_node_" + str(node_number)
                image_file_name = "patch_patient_" + str(patient_number) + "_node_" + str(node_number) + "_x_" + str(x_coord) + "_y_" + str(y_coord) + ".png"
                full_path = "patches/" + image_folder_name + "/" + image_file_name
                img = Image.open(full_path)
                img = img.convert("RGB")
                img = np.array(img)
                imagesData.append(img)
            imagesData = np.array(imagesData)
            targetsData = tf.keras.utils.to_categorical(targetsData, num_classes=2)
            yield(imagesData, targetsData)

            batchStartIndex = batchStartIndex + batchSize
            batchEndIndex = batchEndIndex + batchSize


def generate_mixed_adversarial_data(model, batchSize, data_frame, eps = 8, loss_object = tf.keras.losses.CategoricalCrossentropy()):

    amountOfImages = len(data_frame)

    while(1):
        batchStartIndex = 0
        batchEndIndex = batchSize

        while (batchStartIndex < amountOfImages):
            endIndex = 0
            if(batchEndIndex < amountOfImages):
                endIndex = batchEndIndex
            else:
                endIndex = amountOfImages

            #Load in batch of images and store them in imagesData
            imagesData = []
            adversarialImagesData = []
            targetsData = []
            adversarialTargetsData = []
            for i in range(batchStartIndex,endIndex):    
                data_frame.iloc[i]
                patient_number = "{:03}".format(data_frame.iloc[i, 1])
                node_number = data_frame.iloc[i, 2]
                x_coord = data_frame.iloc[i, 3]
                y_coord = data_frame.iloc[i, 4]
                targetsData.append(data_frame.iloc[i, 5])
                adversarialTargetsData.append(data_frame.iloc[i, 5])
                image_folder_name = "patient_" + str(patient_number) + "_node_" + str(node_number)
                image_file_name = "patch_patient_" + str(patient_number) + "_node_" + str(node_number) + "_x_" + str(x_coord) + "_y_" + str(y_coord) + ".png"
                full_path = "patches/" + image_folder_name + "/" + image_file_name
                img = Image.open(full_path)
                img = img.convert("RGB")
                img = np.array(img)
                imagesData.append(img)
            imagesData = np.array(imagesData)
            targetsData = tf.keras.utils.to_categorical(targetsData, num_classes=2)

            adv_imgs = tf.cast(imagesData, tf.float32)
            with tf.GradientTape() as tape:
                with tf.device("/device:GPU:0"):
                    tape.watch(adv_imgs)
                    preds = model(adv_imgs)
                    loss = loss_object(targetsData, preds)
                    gradient = tape.gradient(loss, adv_imgs)
                    signedGrad = tf.sign(gradient)
            adversarial_images = np.clip((adv_imgs + (signedGrad * eps)).numpy(), 0, 255).astype("uint8")
            adversarialImagesData.append(adversarial_images)

            adversarialImagesData = np.array(adversarialImagesData)
            adversarialImagesData = np.squeeze(adversarialImagesData, axis=0)
            adversarialTargetsData = tf.keras.utils.to_categorical(adversarialTargetsData, num_classes=2)
            mixedImages = np.vstack([imagesData, adversarialImagesData])
            mixedLabels = np.vstack([targetsData, adversarialTargetsData])
            (mixedImages, mixedLabels) = shuffle(mixedImages, mixedLabels)
            yield(mixedImages, mixedLabels)

            batchStartIndex = batchStartIndex + batchSize
            batchEndIndex = batchEndIndex + batchSize


def generate_mixed_noise_data(batchSize, data_frame):

    amountOfImages = len(data_frame)

    while(1):
        batchStartIndex = 0
        batchEndIndex = batchSize

        while (batchStartIndex < amountOfImages):
            endIndex = 0
            if(batchEndIndex < amountOfImages):
                endIndex = batchEndIndex
            else:
                endIndex = amountOfImages

            #Load in batch of images and store them in imagesData
            imagesData = []
            noisy_images_data = []
            targetsData = []
            noisy_targets_data = []
            for i in range(batchStartIndex,endIndex):

                data_frame.iloc[i]
                patient_number = "{:03}".format(data_frame.iloc[i, 1])
                node_number = data_frame.iloc[i, 2]
                x_coord = data_frame.iloc[i, 3]
                y_coord = data_frame.iloc[i, 4]
                targetsData.append(data_frame.iloc[i, 5])
                noisy_targets_data.append(data_frame.iloc[i, 5])
                image_folder_name = "patient_" + str(patient_number) + "_node_" + str(node_number)
                image_file_name = "patch_patient_" + str(patient_number) + "_node_" + str(node_number) + "_x_" + str(x_coord) + "_y_" + str(y_coord) + ".png"
                full_path = "patches/" + image_folder_name + "/" + image_file_name
                img = Image.open(full_path)
                img = img.convert("RGB")
                img = np.array(img)
                imagesData.append(img)

                label = data_frame.iloc[i, 5]
                label = tf.keras.utils.to_categorical(label, num_classes=2)
                label = np.expand_dims(label, axis=0)
                img = np.expand_dims(img, axis=0)
                noisy_img = generate_noisy_image(img)
                noisy_img = np.squeeze(noisy_img, axis=0)
                noisy_images_data.append(noisy_img)

            imagesData = np.array(imagesData)
            noisy_images_data = np.array(noisy_images_data)
            targetsData = tf.keras.utils.to_categorical(targetsData, num_classes=2)
            noisy_targets_data = tf.keras.utils.to_categorical(noisy_targets_data, num_classes=2)
            mixedImages = np.vstack([imagesData, noisy_images_data])
            mixedLabels = np.vstack([targetsData, noisy_targets_data])
            (mixedImages, mixedLabels) = shuffle(mixedImages, mixedLabels)

            yield(mixedImages, mixedLabels)

            batchStartIndex = batchStartIndex + batchSize
            batchEndIndex = batchEndIndex + batchSize


def model_evaluation(modelName, model, data_frame, batch_size, test_data_name):
    x_test = []
    y_test = []

    for i in range(0, len(data_frame)):    
        data_frame.iloc[i]
        patient_number = "{:03}".format(data_frame.iloc[i, 1])
        node_number = data_frame.iloc[i, 2]
        x_coord = data_frame.iloc[i, 3]
        y_coord = data_frame.iloc[i, 4]
        y_test.append(data_frame.iloc[i, 5])
        image_folder_name = "patient_" + str(patient_number) + "_node_" + str(node_number)
        image_file_name = "patch_patient_" + str(patient_number) + "_node_" + str(node_number) + "_x_" + str(x_coord) + "_y_" + str(y_coord) + ".png"
        full_path = "patches/" + image_folder_name + "/" + image_file_name
        img = Image.open(full_path)
        img = img.convert("RGB")
        img = np.array(img)
        x_test.append(img)
    x_test = np.array(x_test)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)  
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    f = open("model_evaluations.txt", "a")
    f.write(modelName + " Has a test loss of: " + str(results[0]) + " and test acc of: " + str(results[1]) + " on " + str(test_data_name) + "\n")
    f.close()
    print(modelName + " Has a test loss and test acc of:", results)
    return results

def extract_activations(model, dataGenerator):
    layer_outputs = [layer.output for layer in model.layers[6:]]
    peek_at_layers = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
    activations = []

    img, msk = dataGenerator.__next__()
    lastConvLayer = peek_at_layers.predict(img)
    out = np.squeeze(np.array(lastConvLayer[0]))
    activations.append(out)
    activations = np.squeeze(activations, axis=0)
    return(np.array(activations))


def representation_shift(act_ref, act_test):

    wass_dist = [wasserstein_distance(act_ref[:, channel], act_test[:, channel]) for channel in range(act_ref.shape[1])]
    return np.asarray(wass_dist).mean()

def measure_representation_shift_exp1(modelName, model_load_path, reference_data, indist_data, outdist_data):

    model = load_model(model_load_path)
    reference_datagen = generate_data(1, reference_data)
    indist_datagen = generate_data(1, indist_data)
    outdist_datagen = generate_data(1, outdist_data)

    activations_ref = extract_activations(model, reference_datagen)
    activations_indist = extract_activations(model, indist_datagen)
    activations_outdist = extract_activations(model, outdist_datagen)

    in_dist_shift = representation_shift(activations_ref, activations_indist)
    out_dist_shift = representation_shift(activations_ref, activations_outdist)

    f = open("model_rep_shifts.txt", "a")
    f.write(modelName + " Has an in distribution representation shift of: " + str(in_dist_shift))
    f.write(modelName + "Has an out of distribution representation shift of:" + str(out_dist_shift) + "\n")
    f.close()
    print(modelName + " Has an in distribution representation shift of: " + str(in_dist_shift))
    print(modelName + "Has an out of distribution representation shift of:" + str(out_dist_shift) + "\n")

def measure_representation_shift_exp2_h4(num_samples, modelName, model_load_path, reference_data, indist_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_5_data):
    model = load_model(model_load_path)
    reference_datagen = generate_data(num_samples, reference_data)
    indist_datagen = generate_data(num_samples, indist_data)
    outdist_datagen_h1 = generate_data(num_samples, hospital_1_data)
    outdist_datagen_h2 = generate_data(num_samples, hospital_2_data)
    outdist_datagen_h3 = generate_data(num_samples, hospital_3_data)
    outdist_datagen_h5 = generate_data(num_samples, hospital_5_data)
    activations_ref = extract_activations(model, reference_datagen)
    activations_indist = extract_activations(model, indist_datagen)
    activations_outdist_h1 = extract_activations(model, outdist_datagen_h1)
    activations_outdist_h2 = extract_activations(model, outdist_datagen_h2)
    activations_outdist_h3 = extract_activations(model, outdist_datagen_h3)
    activations_outdist_h5 = extract_activations(model, outdist_datagen_h5)

    in_dist_shift = representation_shift(activations_ref, activations_indist)
    out_dist_shift_hospital1 = representation_shift(activations_ref, activations_outdist_h1)
    out_dist_shift_hospital2 = representation_shift(activations_ref, activations_outdist_h2)
    out_dist_shift_hospital3 = representation_shift(activations_ref, activations_outdist_h3)
    out_dist_shift_hospital5 = representation_shift(activations_ref, activations_outdist_h5)

    f = open("model_rep_shifts.txt", "a")
    f.write(modelName + " Has an representation shift with Hospital 1 data of: " + str(out_dist_shift_hospital1) + "\n")
    f.write(modelName + " Has an representation shift with Hospital 2 data of: " + str(out_dist_shift_hospital2) + "\n")
    f.write(modelName + " Has an representation shift with Hospital 3 data of: " + str(out_dist_shift_hospital3) + "\n")
    f.write(modelName + " Has an representation shift with Hospital 4 data of: " + str(in_dist_shift) + "\n")
    f.write(modelName + " Has an representation shift with Hospital 5 data of: " + str(out_dist_shift_hospital5) + "\n")
    f.close()
    print(modelName + " Has an in distribution representation shift of: " + str(in_dist_shift))
    print(modelName + " Has an representation shift with Hospital 1 data of: " + str(out_dist_shift_hospital1))
    print(modelName + " Has an representation shift with Hospital 2 data of: " + str(out_dist_shift_hospital2))
    print(modelName + " Has an representation shift with Hospital 3 data of: " + str(out_dist_shift_hospital3))
    print(modelName + " Has an representation shift with Hospital 5 data of: " + str(out_dist_shift_hospital5))

def measure_representation_shift_exp2_h5(num_samples, modelName, model_load_path, reference_data, indist_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_4_data):

    model = load_model(model_load_path)
    reference_datagen = generate_data(num_samples, reference_data)
    indist_datagen = generate_data(num_samples, indist_data)
    outdist_datagen_h1 = generate_data(num_samples, hospital_1_data)
    outdist_datagen_h2 = generate_data(num_samples, hospital_2_data)
    outdist_datagen_h3 = generate_data(num_samples, hospital_3_data)
    outdist_datagen_h4 = generate_data(num_samples, hospital_4_data)

    activations_ref = extract_activations(model, reference_datagen)
    activations_indist = extract_activations(model, indist_datagen)
    activations_outdist_h1 = extract_activations(model, outdist_datagen_h1)
    activations_outdist_h2 = extract_activations(model, outdist_datagen_h2)
    activations_outdist_h3 = extract_activations(model, outdist_datagen_h3)
    activations_outdist_h4 = extract_activations(model, outdist_datagen_h4)

    in_dist_shift = representation_shift(activations_ref, activations_indist)
    out_dist_shift_hospital1 = representation_shift(activations_ref, activations_outdist_h1)
    out_dist_shift_hospital2 = representation_shift(activations_ref, activations_outdist_h2)
    out_dist_shift_hospital3 = representation_shift(activations_ref, activations_outdist_h3)
    out_dist_shift_hospital4 = representation_shift(activations_ref, activations_outdist_h4)

    f = open("model_rep_shifts.txt", "a")
    f.write(modelName + " Has an representation shift with Hospital 1 data of: " + str(out_dist_shift_hospital1) + "\n")
    f.write(modelName + " Has an representation shift with Hospital 2 data of: " + str(out_dist_shift_hospital2) + "\n")
    f.write(modelName + " Has an representation shift with Hospital 3 data of: " + str(out_dist_shift_hospital3) + "\n")
    f.write(modelName + " Has an representation shift with Hospital 4 data of: " + str(out_dist_shift_hospital4) + "\n")
    f.write(modelName + " Has an representation shift with Hospital 5 data of: " + str(in_dist_shift) + "\n")
    f.close()
    print(modelName + " Has an in distribution representation shift of: " + str(in_dist_shift))
    print(modelName + " Has an representation shift with Hospital 1 data of: " + str(out_dist_shift_hospital1))
    print(modelName + " Has an representation shift with Hospital 2 data of: " + str(out_dist_shift_hospital2))
    print(modelName + " Has an representation shift with Hospital 3 data of: " + str(out_dist_shift_hospital3))
    print(modelName + " Has an representation shift with Hospital 5 data of: " + str(out_dist_shift_hospital4))

def run_experiment_1_standard(batch_size, model_name, training_data, validation_data, test_data):

    for i in range(1,11):
        shuffled_training_data = training_data.sample(frac=1)
        shuffled_validation_data = validation_data.sample(frac=1)
        shuffled_test_data = test_data.sample(frac=1)
        shuffled_training_data_1 = shuffled_training_data.iloc[0:89697]
        shuffled_training_data_2 = shuffled_training_data.iloc[89697:179393]

        ## Standard Training
        model = create_DenseNet121()
        batch_size = batch_size
        steps_epoch = len(shuffled_training_data) // batch_size
        val_steps_epoch = len(shuffled_validation_data) // batch_size

        standard_training_datagen = generate_data(batch_size, shuffled_training_data)

        standard_validation_datagen = generate_data(batch_size, shuffled_validation_data)

        with tf.device("/device:GPU:0"):
            hist = model.fit(standard_training_datagen, steps_per_epoch = steps_epoch, epochs=5, verbose=1, validation_data=standard_validation_datagen, validation_steps=val_steps_epoch)
            model.save('models/' + model_name + str(i) + '.hdf5')

        model_evaluation(model_name + str(i), model, shuffled_test_data, 32, "hospital 5 test data")
        model_load_path = 'models/' + model_name + str(i) + '.hdf5'
        measure_representation_shift_exp1(model_name + str(i), model_load_path, shuffled_training_data_1, shuffled_training_data_2, shuffled_test_data)

def run_experiment_1_adversarial(batch_size, model_name, training_data, validation_data, test_data):
    
    for i in range(1,11):
        shuffled_training_data = training_data.sample(frac=1)
        shuffled_training_data_1 = shuffled_training_data.iloc[0:89697]
        shuffled_training_data_2 = shuffled_training_data.iloc[89697:179393]
        shuffled_validation_data = validation_data.sample(frac=1)
        shuffled_test_data = test_data.sample(frac=1)

        model = create_DenseNet121()
        mixed_batch_size = int(batch_size / 2)
        steps_epoch = len(shuffled_training_data) // mixed_batch_size
        val_steps_epoch = len(shuffled_validation_data) // mixed_batch_size

        adversarial_training_datagen = generate_mixed_adversarial_data(model, mixed_batch_size, shuffled_training_data)

        adversarial_validation_datagen =  generate_mixed_adversarial_data(model, mixed_batch_size, shuffled_validation_data)

        with tf.device("/device:GPU:0"):
            hist = model.fit(adversarial_training_datagen, steps_per_epoch = steps_epoch, epochs=5, verbose=1, validation_data=adversarial_validation_datagen, validation_steps=val_steps_epoch)
            model.save('models/' + model_name + str(i) + '.hdf5')

        model_evaluation(model_name + str(i), model, shuffled_test_data, 32, "hospital 5 test data")
        model_load_path = 'models/' + model_name + str(i) + '.hdf5'
        measure_representation_shift_exp1(model_name + str(i), model_load_path, shuffled_training_data_1, shuffled_training_data_2, shuffled_test_data)

def run_experiment_1_noise(batch_size, model_name, training_data, validation_data, test_data):

    for i in range(1,11):
        shuffled_training_data = training_data.sample(frac=1)
        shuffled_validation_data = validation_data.sample(frac=1)
        shuffled_test_data = test_data.sample(frac=1)
        shuffled_training_data_1 = shuffled_training_data.iloc[0:89697]
        shuffled_training_data_2 = shuffled_training_data.iloc[89697:179393]

        model = create_DenseNet121()
        mixed_batch_size = int(batch_size / 2)
        steps_epoch = len(shuffled_training_data) // mixed_batch_size
        val_steps_epoch = len(shuffled_validation_data) // mixed_batch_size

        noise_training_datagen = generate_mixed_noise_data(mixed_batch_size, shuffled_training_data)

        noise_validation_datagen =  generate_mixed_noise_data(mixed_batch_size, shuffled_validation_data)

        with tf.device("/device:GPU:0"):
            hist = model.fit(noise_training_datagen, steps_per_epoch = steps_epoch, epochs=5, verbose=1, validation_data=noise_validation_datagen, validation_steps=val_steps_epoch)
            model.save('models/' + model_name + str(i) + '.hdf5')
            
        model_evaluation(model_name + str(i), model, shuffled_test_data, 32, "hospital 5 test data")
        model_load_path = 'models/' + model_name + str(i) + '.hdf5'
        measure_representation_shift_exp1(model_name + str(i), model_load_path, shuffled_training_data_1, shuffled_training_data_2, shuffled_test_data)

def run_experiment_2_standard_h4(batch_size, model_name, hospital_4_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_5_data):

    for i in range(1,11):
        shuffled_hospital_4_data = hospital_4_data.sample(frac=1)
        hospital_4_data_training = shuffled_hospital_4_data.iloc[0:103871]
        hospital_4_data_validation = shuffled_hospital_4_data.iloc[103871:129838]
        ## Standard Training
        model = create_DenseNet121()
        batch_size = batch_size
        steps_epoch = len(hospital_4_data_training) // batch_size
        val_steps_epoch = len(hospital_4_data_validation) // batch_size

        standard_training_datagen = generate_data(batch_size, hospital_4_data_training)

        standard_validation_datagen = generate_data(batch_size, hospital_4_data_validation)

        with tf.device("/device:GPU:0"):
            hist = model.fit(standard_training_datagen, steps_per_epoch = steps_epoch, epochs=5, verbose=1, validation_data=standard_validation_datagen, validation_steps=val_steps_epoch)
            model.save('models/' + model_name + str(i) + '.hdf5')

        f = open("model_evaluations.txt", "a")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 1 data of:")
        model_evaluation(model_name + str(i), model, hospital_1_data, 32, "hospital 1 data")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 2 data of:")
        model_evaluation(model_name + str(i), model, hospital_2_data, 32, "hospital 2 data")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 3 data of:")
        model_evaluation(model_name + str(i), model, hospital_3_data, 32, "hospital 3 data")       
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 5 data of:")
        model_evaluation(model_name + str(i), model, hospital_5_data, 32, "hospital 5 data") 
        f.close()       

        model_load_path = 'models/' + model_name + str(i) + '.hdf5'
        measure_representation_shift_exp2_h4(1000, model_name + str(i), model_load_path, hospital_4_data_training, hospital_4_data_validation, hospital_1_data, hospital_2_data, hospital_3_data, hospital_5_data)

def run_experiment_2_adversarial_h4(batch_size, model_name, hospital_4_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_5_data):

    for i in range(1,11):
        shuffled_hospital_4_data = hospital_4_data.sample(frac=1)
        hospital_4_data_training = shuffled_hospital_4_data.iloc[0:103871]
        hospital_4_data_validation = shuffled_hospital_4_data.iloc[103871:129838]

        ## NoiseTraining
        model = create_DenseNet121()
        mixed_batch_size = int(batch_size / 2)
        steps_epoch = len(hospital_4_data_training) // mixed_batch_size
        val_steps_epoch = len(hospital_4_data_validation) // mixed_batch_size

        standard_training_datagen = generate_mixed_adversarial_data(model, mixed_batch_size, hospital_4_data_training)

        standard_validation_datagen = generate_mixed_adversarial_data(model, mixed_batch_size, hospital_4_data_validation)

        with tf.device("/device:GPU:0"):
            hist = model.fit(standard_training_datagen, steps_per_epoch = steps_epoch, epochs=5, verbose=1, validation_data=standard_validation_datagen, validation_steps=val_steps_epoch)
            model.save('models/' + model_name + str(i) + '.hdf5')

        f = open("model_evaluations.txt", "a")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 1 data of:")
        model_evaluation(model_name + str(i), model, hospital_1_data, 32, "hospital 1 data")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 2 data of:")
        model_evaluation(model_name + str(i), model, hospital_2_data, 32, "hospital 2 data")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 3 data of:")
        model_evaluation(model_name + str(i), model, hospital_3_data, 32, "hospital 3 data")      
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 5 data of:")
        model_evaluation(model_name + str(i), model, hospital_5_data, 32, "hospital 5 data") 
        f.close()       

        model_load_path = 'models/' + model_name + str(i) + '.hdf5'
        measure_representation_shift_exp2_h4(1000, model_name + str(i), model_load_path, hospital_4_data_training, hospital_4_data_validation, hospital_1_data, hospital_2_data, hospital_3_data, hospital_5_data)

def run_experiment_2_noise_h4(batch_size, model_name, hospital_4_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_5_data):

    for i in range(1,11):
        shuffled_hospital_4_data = hospital_4_data.sample(frac=1)
        hospital_4_data_training = shuffled_hospital_4_data.iloc[0:103871]
        hospital_4_data_validation = shuffled_hospital_4_data.iloc[103871:129838]

        ## NoiseTraining
        model = create_DenseNet121()
        mixed_batch_size = int(batch_size / 2)
        steps_epoch = len(hospital_4_data_training) // mixed_batch_size
        val_steps_epoch = len(hospital_4_data_validation) // mixed_batch_size

        noise_training_datagen = generate_mixed_noise_data(mixed_batch_size, hospital_4_data_training)

        noise_validation_datagen = generate_mixed_noise_data(mixed_batch_size, hospital_4_data_validation)

        with tf.device("/device:GPU:0"):
            hist = model.fit(noise_training_datagen, steps_per_epoch = steps_epoch, epochs=5, verbose=1, validation_data=noise_validation_datagen, validation_steps=val_steps_epoch)
            model.save('models/' + model_name + str(i) + '.hdf5')

        f = open("model_evaluations.txt", "a")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 1 data of:")
        model_evaluation(model_name + str(i), model, hospital_1_data, 32, "hospital 1 data")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 2 data of:")
        model_evaluation(model_name + str(i), model, hospital_2_data, 32, "hospital 2 data")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 3 data of:")
        model_evaluation(model_name + str(i), model, hospital_3_data, 32, "hospital 3 data")        
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 5 data of:")
        model_evaluation(model_name + str(i), model, hospital_5_data, 32, "hospital 5 data") 
        f.close()       

        model_load_path = 'models/' + model_name + str(i) + '.hdf5'
        measure_representation_shift_exp2_h4(1000, model_name + str(i), model_load_path, hospital_4_data_training, hospital_4_data_validation, hospital_1_data, hospital_2_data, hospital_3_data, hospital_5_data)

def run_experiment_2_standard_h5(batch_size, model_name, hospital_5_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_4_data):

    for i in range(1,11):
        shuffled_hospital_5_data = hospital_5_data.sample(frac=1)
        hospital_5_data_training = shuffled_hospital_5_data.iloc[0:117378]
        hospital_5_data_validation = shuffled_hospital_5_data.iloc[117378:146722]

        ## Standard Training
        model = create_DenseNet121()
        batch_size = batch_size
        steps_epoch = len(hospital_5_data_training) // batch_size
        val_steps_epoch = len(hospital_5_data_validation) // batch_size

        standard_training_datagen = generate_data(batch_size, hospital_5_data_training)

        standard_validation_datagen = generate_data(batch_size, hospital_5_data_validation)

        with tf.device("/device:GPU:0"):
            hist = model.fit(standard_training_datagen, steps_per_epoch = steps_epoch, epochs=5, verbose=1, validation_data=standard_validation_datagen, validation_steps=val_steps_epoch)
            model.save('models/' + model_name + str(i) + '.hdf5')

        f = open("model_evaluations.txt", "a")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 1 data of:")
        model_evaluation(model_name + str(i), model, hospital_1_data, 32, "hospital 1 data")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 2 data of:")
        model_evaluation(model_name + str(i), model, hospital_2_data, 32, "hospital 2 data")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 3 data of:")
        model_evaluation(model_name + str(i), model, hospital_3_data, 32, "hospital 3 data")  
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 4 data of:")
        model_evaluation(model_name + str(i), model, hospital_4_data, 32, "hospital 4 data")       
        f.close()       

        model_load_path = 'models/' + model_name + str(i) + '.hdf5'
        measure_representation_shift_exp2_h5(1000, model_name + str(i), model_load_path, hospital_5_data_training, hospital_5_data_validation, hospital_1_data, hospital_2_data, hospital_3_data, hospital_4_data)

def run_experiment_2_adversarial_h5(batch_size, model_name, hospital_5_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_4_data):

    for i in range(1,11):
        shuffled_hospital_5_data = hospital_5_data.sample(frac=1)
        hospital_5_data_training = shuffled_hospital_5_data.iloc[0:117378]
        hospital_5_data_validation = shuffled_hospital_5_data.iloc[117378:146722]

        ## NoiseTraining
        model = create_DenseNet121()
        mixed_batch_size = int(batch_size / 2)
        steps_epoch = len(hospital_5_data_training) // mixed_batch_size
        val_steps_epoch = len(hospital_5_data_validation) // mixed_batch_size

        adv_training_datagen = generate_mixed_adversarial_data(model, mixed_batch_size, hospital_5_data_training)

        adv_validation_datagen = generate_mixed_adversarial_data(model, mixed_batch_size, hospital_5_data_validation)

        with tf.device("/device:GPU:0"):
            hist = model.fit(adv_training_datagen, steps_per_epoch = steps_epoch, epochs=5, verbose=1, validation_data=adv_validation_datagen, validation_steps=val_steps_epoch)
            model.save('models/' + model_name + str(i) + '.hdf5')

        f = open("model_evaluations.txt", "a")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 1 data of:")
        model_evaluation(model_name + str(i), model, hospital_1_data, 32, "hospital 1 data")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 2 data of:")
        model_evaluation(model_name + str(i), model, hospital_2_data, 32, "hospital 2 data")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 3 data of:")
        model_evaluation(model_name + str(i), model, hospital_3_data, 32, "hospital 3 data")  
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 4 data of:")
        model_evaluation(model_name + str(i), model, hospital_4_data, 32, "hospital 4 data")       
        f.close()       

        model_load_path = 'models/' + model_name + str(i) + '.hdf5'
        measure_representation_shift_exp2_h5(1000, model_name + str(i), model_load_path, hospital_5_data_training, hospital_5_data_validation, hospital_1_data, hospital_2_data, hospital_3_data, hospital_4_data)

def run_experiment_2_noise_h5(batch_size, model_name, hospital_5_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_4_data):

    for i in range(1,11):
        shuffled_hospital_5_data = hospital_5_data.sample(frac=1)
        hospital_5_data_training = shuffled_hospital_5_data.iloc[0:117378]
        hospital_5_data_validation = shuffled_hospital_5_data.iloc[117378:146722]

        ## NoiseTraining
        model = create_DenseNet121()
        mixed_batch_size = int(batch_size / 2)
        steps_epoch = len(hospital_5_data_training) // mixed_batch_size
        val_steps_epoch = len(hospital_5_data_validation) // mixed_batch_size

        noise_training_datagen = generate_mixed_noise_data(mixed_batch_size, hospital_5_data_training)

        noise_validation_datagen = generate_mixed_noise_data(mixed_batch_size, hospital_5_data_validation)

        with tf.device("/device:GPU:0"):
            hist = model.fit(noise_training_datagen, steps_per_epoch = steps_epoch, epochs=5, verbose=1, validation_data=noise_validation_datagen, validation_steps=val_steps_epoch)
            model.save('models/' + model_name + str(i) + '.hdf5')

        f = open("model_evaluations.txt", "a")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 1 data of:")
        model_evaluation(model_name + str(i), model, hospital_1_data, 32, "hospital 1 data")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 2 data of:")
        model_evaluation(model_name + str(i), model, hospital_2_data, 32, "hospital 2 data")
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 3 data of:")
        model_evaluation(model_name + str(i), model, hospital_3_data, 32, "hospital 3 data")  
        print(model_name + str(i) + "has a test loss and accuracy on Hospital 4 data of:")
        model_evaluation(model_name + str(i), model, hospital_4_data, 32, "hospital 4 data")       
        f.close()       

        model_load_path = 'models/' + model_name + str(i) + '.hdf5'
        measure_representation_shift_exp2_h5(1000, model_name + str(i), model_load_path, hospital_5_data_training, hospital_5_data_validation, hospital_1_data, hospital_2_data, hospital_3_data, hospital_4_data)

   
data_frame = pd.read_csv('metadata.csv')

hospital_1_data = data_frame[data_frame["center"] == 0]
shuffled_hospital_1_data = hospital_1_data.sample(frac=1)
hospital_1_data_training = shuffled_hospital_1_data.iloc[0:47549]
hospital_1_data_validation = shuffled_hospital_1_data.iloc[47549:53492]
hospital_1_data_test = shuffled_hospital_1_data.iloc[53492:59435]

hospital_2_data = data_frame[data_frame["center"] == 1]
shuffled_hospital_2_data = hospital_2_data.sample(frac=1)

hospital_3_data = data_frame[data_frame["center"] == 2]
shuffled_hospital_3_data = hospital_3_data.sample(frac=1)

hospital_4_data = data_frame[data_frame["center"] == 3]
shuffled_hospital_4_data = hospital_4_data.sample(frac=1)
hospital_4_data_training = shuffled_hospital_4_data.iloc[0:103871]
hospital_4_data_validation = shuffled_hospital_4_data.iloc[103871:129838]

hospital_5_data = data_frame[data_frame["center"] == 4]
shuffled_hospital_5_data = hospital_5_data.sample(frac=1)

hospital_5_data_training = shuffled_hospital_5_data.iloc[0:117378]
hospital_5_data_validation = shuffled_hospital_5_data.iloc[117378:146722]

#To run original experiment from WILDs Benchmark paper:
training_data = data_frame[(data_frame["center"] == 0) | (data_frame["center"] == 1) | (data_frame["center"] == 2)]
shuffled_training_data = training_data.sample(frac=1)
shuffled_training_data_1 = shuffled_training_data.iloc[0:89697]
shuffled_training_data_2 = shuffled_training_data.iloc[89697:179393]

validation_data = data_frame[data_frame["center"] == 3]
shuffled_validation_data = validation_data.sample(frac=1)

test_data = data_frame[data_frame["center"] == 4]
shuffled_test_data = test_data.sample(frac=1)        
        
run_experiment_1_standard(32, "std_trained_hospital123_", training_data, validation_data, test_data)
run_experiment_1_adversarial(32, "adv_trained_hospital123_", training_data, validation_data, test_data)
run_experiment_1_noise(32, "noise_trained_hospital123_", training_data, validation_data, test_data)
run_experiment_2_standard_h4(32, "std_trained_hospital4_", hospital_4_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_5_data)
run_experiment_2_adversarial_h4(32, "adv_trained_hospital4_", hospital_4_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_5_data)
run_experiment_2_noise_h4(32, "noise_trained_hospital4_", hospital_4_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_5_data)
run_experiment_2_standard_h5(32, "std_trained_hospital5_", hospital_5_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_4_data)
run_experiment_2_adversarial_h5(32, "adv_trained_hospital5_", hospital_5_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_4_data)
run_experiment_2_noise_h5(32, "noise_trained_hospital5_", hospital_5_data, hospital_1_data, hospital_2_data, hospital_3_data, hospital_4_data)
