import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

np.random.seed(100)
tf.random.set_random_seed(94)
number_train_epochs = 20000


house = pd.read_pickle("./data/preprocessed_house_nonnormalized.pkl")

# Make simple Preprocessing split
X_train, X_test, y_train, y_test = train_test_split(house.loc[:, house.columns != "winner"], house["winner"], test_size = 0.1, random_state = 100)

# Basic Model
model = keras.Sequential([keras.layers.Dense(40, activation=tf.nn.sigmoid),
                          keras.layers.Dense(20, activation=tf.nn.sigmoid),
                          keras.layers.Dense(2, activation=tf.nn.softmax)])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# Perform Early Stopping Approach

# 1st iteration
model.fit(X_train.to_numpy(), y_train.to_numpy(), epochs=1)
train_loss, train_acc = model.evaluate(X_train.to_numpy(), y_train.to_numpy())
test_loss, test_acc = model.evaluate(X_test.to_numpy(), y_test.to_numpy())
loss_df = pd.DataFrame({'Training Loss':[train_loss],'Testing Loss':[test_loss]})

model.save("./models/simple_nonnormalized_%s.h5" % number_train_epochs)
optimal_model_test_loss = test_loss
optimal_model_epoch_num = 0

for i in range(1, number_train_epochs): # Already performed 0th iteration
    # Backward propagate once, adjust weights
    model.fit(X_train.to_numpy(), y_train.to_numpy(), epochs=1)
    
    # Forward propagate to calculate new errors on training and testing
    train_loss, train_acc = model.evaluate(X_train.to_numpy(), y_train.to_numpy())
    test_loss, test_acc = model.evaluate(X_test.to_numpy(), y_test.to_numpy())
    new_loss = pd.DataFrame({'Training Loss':[train_loss],'Testing Loss':[test_loss]})
    loss_df = pd.concat([loss_df,new_loss]).reset_index(drop=True)
    
    if (test_loss < optimal_model_test_loss):
        print("Updating Optimal model at %dth iteration" % i)
        model.save("./models/simple_nonnormalized_%s.h5" % number_train_epochs)
        optimal_model_test_loss = test_loss
        optimal_model_epoch_num = i

# Plot the Training and Testing Errors over number of epochs
epochs = tuple(range(number_train_epochs))

fig = plt.figure()

fig.suptitle("Loss on Non-Normalized Data over Number of Propagations (Epochs)")

plt.subplot(1, 2, 1)
plt.plot(epochs, loss_df["Training Loss"])
plt.axvline(x=optimal_model_epoch_num, color = 'black')
plt.title('Training Loss over %d Epochs' % number_train_epochs)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, loss_df["Testing Loss"], color = 'r')
plt.axvline(x=optimal_model_epoch_num, color = 'black')
plt.title('Testing Loss over %d Epochs' % number_train_epochs)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')

plt.savefig("./plots/Simple_Loss_Non_Normalized_%s.png" % number_train_epochs)
plt.show()


# Examine Error
optimal_model = keras.models.load_model("./models/simple_nonnormalized_%s.h5" % number_train_epochs)

test_loss, test_acc = optimal_model.evaluate(X_test.to_numpy(), y_test.to_numpy())
print('Quick testing accuracy:', test_acc)

# Confusion Matricies
classify_pred = lambda x: x > 0.5

test_predictions = optimal_model.predict(X_test.to_numpy())
y_test_pred = classify_pred(test_predictions[:,1])
test_cm = confusion_matrix(y_test.to_numpy(), y_test_pred)
print("Testing Confustion Matrix:\n", test_cm)

train_predictions = optimal_model.predict(X_train.to_numpy())
y_train_pred = classify_pred(train_predictions[:,1])
train_cm = confusion_matrix(y_train.to_numpy(), y_train_pred)
print("Training Confustion Matrix:\n", train_cm)


# Plot Predictions and actual results on a 2D plot of Total Contributions vs Operating Expenditures
plt.clf()
fig = plt.figure()
fig.suptitle("Real Election Outcomes vs Simple Predicted Model of Simple Test Set")

plt.subplot(1, 2, 1)
plt.scatter(X_test["ope_exp"][y_test], X_test["net_con"][y_test], alpha = 0.5, label = "Election Win")
plt.scatter(X_test["ope_exp"][~y_test], X_test["net_con"][~y_test], color = 'r', alpha = 0.5, label = "Election Loss")
plt.title('Actual Testing Set')
plt.xlabel('Operating Expenditures')
plt.ylabel('Net Contributions')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test["ope_exp"][y_test_pred], X_test["net_con"][y_test_pred], alpha = 0.5, label = "Predicted Election Win")
plt.scatter(X_test["ope_exp"][~y_test_pred], X_test["net_con"][~y_test_pred], color = 'r', alpha = 0.5, label = "Predicted Election Loss")
plt.title('Neural Network Predictions of Testing Set')
plt.xlabel('Operating Expenditures')
plt.ylabel('Net Contributions')
plt.legend()

plt.savefig("./plots/Simple_Classifications_Non_Normalized_%s.png" % number_train_epochs)

incorrect_pred = np.array(y_test_pred != y_test)

plt.clf()
plt.scatter(X_test["ope_exp"][incorrect_pred], X_test["net_con"][incorrect_pred], alpha = 0.5, color = 'r')
plt.title('Misclassified Data Points by Non Normalized ANN of Simple Test Set')
plt.xlabel('Operating Expenditures')
plt.ylabel('Net Contributions')
plt.show()
plt.savefig("./plots/Simple_Misclassified_Non_Normalized_%s.png" % number_train_epochs)
