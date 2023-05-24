from tensorflow.keras.models import Sequential, load_model
from keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU
import matplotlib.pyplot as plt
from keras.layers import Dropout, BatchNormalization
import random



from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np


# Load the images and parameters

width = 320
height = 440

image_list = []
param_list = []
train_indices = []
temp_indices = []

# Process the images and parameters from each folder
'''
big_distortion_big_translation
distortion+translation
nightmare
one_corner_out
only_distortion
only_translation
two_corner_out
'''

for folder_name in os.listdir("datasets/Testset"):
    print(folder_name)
    if folder_name not in ["big_distortion_big_translation", "nightmare", "two_corner_out"]:
        folder_path = os.path.join("datasets/Testset", folder_name)
        files = os.listdir(folder_path)
        num_files = len(files)
        num_train_files = num_files // 2
        
        train_files = random.sample(files, num_train_files)
        temp_files = [f for f in files if f not in train_files]

        for i, file_name in enumerate(files):
            img = cv2.imread(os.path.join(folder_path, file_name))
            
            image_list.append(img)
            
            param = file_name[:-4].split()
            param = [float(p) for p in param]
            param_list.append(param)

            if file_name in train_files:
                train_indices.append(len(image_list) - 1)
            else:
                temp_indices.append(len(image_list) - 1)

# Convert lists into numpy arrays
X = np.array(image_list)
Y = np.array(param_list)

# Create train and temp lists
X_train = X[train_indices]
Y_train = Y[train_indices]

X_temp = X[temp_indices]
Y_temp = Y[temp_indices]

X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)






minimum_loss = 10000000


def create_model_1():
    model = Sequential()

    # Add the first convolutional layer with 8 filters, a kernel size of 3x3, and ReLU activation
    model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(width, height, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))

    # Flatten the output of the third convolutional layer
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    # Add a fully connected layer with 3 neurons for the output (focal length, camera position x, camera position y, camera position z, camera angle yaw, camera angle pitch, camera angle roll)
    model.add(Dense(7))

    model.compile(loss='mse', optimizer='adam')
    return model

def create_model_2():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(width, height, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (7, 7), activation='relu'))
    model.add(MaxPooling2D((2, 2)))


    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(7))

    model.compile(loss='mse', optimizer='adam')

    return model

def run_once(ep, bat):
    mse_loss = []

    model = create_model_1()
    history = model.fit(X_train, Y_train, epochs=ep, batch_size=bat, validation_data=(X_val, Y_val))
    test_loss = model.evaluate(X_test, Y_test)

    # Make predictions for the test set
    Y_pred = model.predict(X_test)

    # Calculate individual losses for each Y parameter
    individual_losses = np.mean((Y_test - Y_pred)**2, axis=0)

    # Create a 1x2 grid of subplots for the loss graph
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i in range(7):
        row = i // 4
        col = i % 4

        # Plot the training and validation loss vs. epochs for the i-th Y parameter
        axes[row, col].plot(history.history['loss'], label='train')
        axes[row, col].plot(history.history['val_loss'], label='val')
        axes[row, col].set_xlabel('Epochs')
        axes[row, col].set_ylabel('Loss')
        axes[row, col].legend()
        axes[row, col].set_title(f'Loss vs. Epochs (Y{i+1}, MSE: {individual_losses[i]:.4f})')

    # Remove the last (unused) subplot
    fig.delaxes(axes[1, 3])

    plt.show()

    mse_loss.append(test_loss)

    return test_loss, model, history  

file = open("training information.txt", "w")
for i in range(0, 1):
    epoch = 20
    batch = 16
    file.write("   $$$$$$")
    test_loss, model, history = run_once(epoch, batch)

    if test_loss < minimum_loss:
        print(f"{i}th iteration ******************************")

        file.write(f"   {test_loss}$$$$$${epoch}$$$$$$${batch}    ")

        minimum_loss = test_loss

        model.save('loss.h5')


file.close()


loaded_model = load_model('loss.h5')

# Use the loaded model to make predictions on the test dataset
Y_pred = loaded_model.predict(X_test)

# Evaluate the predictions using some metric (e.g. mean squared error)
mse_loss_params = []
for i in range(0, 7):
    mse_loss_i = np.mean((Y_pred[:, i] - Y_test[:, i])**2)
    mse_loss_params.append(mse_loss_i)
    print(f"MSE loss for parameter {i}: {mse_loss_i}")

print("Final MSE loss for each set of parameters:", mse_loss_params)




#try to increase the kernel size and decrease the number of layers as the depth increase
#visual validation dataset



# raw image, data image,  model reconstruction, difference image
#show differnt kind of distortion 



#break the milestone objectives into different small activities