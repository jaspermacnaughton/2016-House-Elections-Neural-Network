import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

np.random.seed(272)
tf.random.set_random_seed(86)

n_folds = 5
max_iterations = 300
converge_iterations = 50

def Early_Stop_Train(model, X_train, y_train, X_test, y_test, max_iterations, converge_iterations, model_filename):
    # Take the model structure in as model
    
    # 1st iteration
    model.fit(X_train, y_train, epochs = 1)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    model.save(model_filename)
    optimal_model_test_acc = test_acc
    optimal_model_epoch_num = 0
    #optimal_model = model # Initialize optimal model
    for i in range(1, max_iterations): # Already performed 0th iteration
        # Backward propagate once, adjust weights
        model.fit(X_train, y_train, epochs = 1)
        
        # Just forward propagate to calculate testing set error
        test_loss, test_acc = model.evaluate(X_test, y_test)
        
        if (test_acc > optimal_model_test_acc):
            model.save(model_filename)
            optimal_model_test_acc = test_acc
            optimal_model_epoch_num = i
        
        if (i > (optimal_model_epoch_num + converge_iterations)):
            # Then we have gone "long enough" to return model
            break
    # Models already saved to filesystem- no need to reload
    #optimal_model = keras.models.load_model(model_filename)
    return (optimal_model_test_acc, i)




def Initialize_Models():
    
    
    # Three Sigmoid Models to compare how they different numbers of layers perform
    one_layer_sigmoid = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation = tf.nn.sigmoid, input_shape = (40,)),
            tf.keras.layers.Dense(10, activation = tf.nn.sigmoid),
            tf.keras.layers.Dense(2, activation = tf.nn.softmax)
            ])
    one_layer_sigmoid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    two_layer_sigmoid = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation = tf.nn.sigmoid, input_shape = (40,)),
            tf.keras.layers.Dense(30, activation = tf.nn.sigmoid),
            tf.keras.layers.Dense(15, activation = tf.nn.sigmoid),
            tf.keras.layers.Dense(2, activation = tf.nn.softmax) 
            ])
    two_layer_sigmoid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    three_layer_sigmoid = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation = tf.nn.sigmoid, input_shape = (40,)),
            tf.keras.layers.Dense(40, activation = tf.nn.sigmoid),
            tf.keras.layers.Dense(25, activation = tf.nn.sigmoid),
            tf.keras.layers.Dense(10, activation = tf.nn.sigmoid),
            tf.keras.layers.Dense(2, activation = tf.nn.softmax)
            ])
    three_layer_sigmoid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    three_layer_sigmoid = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation = tf.nn.sigmoid, input_shape = (40,)),
            tf.keras.layers.Dense(40, activation = tf.nn.sigmoid),
            tf.keras.layers.Dense(25, activation = tf.nn.sigmoid),
            tf.keras.layers.Dense(10, activation = tf.nn.sigmoid),
            tf.keras.layers.Dense(2, activation = tf.nn.softmax)
            ])
    three_layer_sigmoid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
    # Models with different activations, optimizer, and loss to compare how these perform
    relu_activation = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation = tf.nn.relu, input_shape = (40,)),
            tf.keras.layers.Dense(10, activation = tf.nn.relu),
            tf.keras.layers.Dense(2, activation = tf.nn.softmax)
            ])
    relu_activation.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    RMSProp_optimizer = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation = tf.nn.sigmoid, input_shape = (40,)),
            tf.keras.layers.Dense(10, activation = tf.nn.sigmoid),
            tf.keras.layers.Dense(2, activation = tf.nn.softmax)
            ])
    RMSProp_optimizer.compile(optimizer='RMSProp', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    deep_relu_activation = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation = tf.nn.relu, input_shape = (40,)),
            tf.keras.layers.Dense(40, activation = tf.nn.relu),
            tf.keras.layers.Dense(60, activation = tf.nn.relu),
            tf.keras.layers.Dense(40, activation = tf.nn.relu),
            tf.keras.layers.Dense(20, activation = tf.nn.relu),
            tf.keras.layers.Dense(10, activation = tf.nn.relu),
            tf.keras.layers.Dense(5, activation = tf.nn.relu),
            tf.keras.layers.Dense(2, activation = tf.nn.softmax)
            ])
    deep_relu_activation.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return (one_layer_sigmoid, two_layer_sigmoid, three_layer_sigmoid, relu_activation, RMSProp_optimizer, deep_relu_activation)



# Select Normalized data as it resulted in a smoother, quicker optimization
house = pd.read_pickle("./data/preprocessed_house.pkl")
X = house.loc[:, house.columns != "winner"].to_numpy()
y = house["winner"].to_numpy()
folds = KFold(n_splits = n_folds, random_state = 24)

# Estimate accuracy by applying K-Fold Cross Validation across 
i = 1
performance = {
        "one_layer_sigmoid": [],
        "two_layer_sigmoid": [],
        "three_layer_sigmoid": [],
        "relu_activation": [],
        "RMSProp_optimizer": [],
        "deep_relu_activation": []
        }
iterations = {
        "one_layer_sigmoid": [],
        "two_layer_sigmoid": [],
        "three_layer_sigmoid": [],
        "relu_activation": [],
        "RMSProp_optimizer": [],
        "deep_relu_activation": []
        }


for train_index, test_index in folds.split(house):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize models to start from scratch at each training
    one_layer_sigmoid, two_layer_sigmoid, three_layer_sigmoid, relu_activation, RMSProp_optimizer, deep_relu_activation = Initialize_Models()

    # ---------------------- One layer sigmoid ----------------------
    one_layer_sigmoid_test_acc, one_layer_sigmoid_iterations = Early_Stop_Train(
            one_layer_sigmoid, X_train, y_train, X_test, y_test, 
            max_iterations, converge_iterations, "./models/cv/one_layer_sigmoid_%d.h5" % i)
    performance["one_layer_sigmoid"].append(one_layer_sigmoid_test_acc)
    iterations["one_layer_sigmoid"].append(one_layer_sigmoid_iterations)
    
    # ---------------------- Two layer sigmoid ----------------------
    two_layer_sigmoid_test_acc, two_layer_sigmoid_iterations = Early_Stop_Train(
            two_layer_sigmoid, X_train, y_train, X_test, y_test, 
            max_iterations, converge_iterations, "./models/cv/two_layer_sigmoid_%d.h5" % i)
    performance["two_layer_sigmoid"].append(two_layer_sigmoid_test_acc)
    iterations["two_layer_sigmoid"].append(two_layer_sigmoid_iterations)
    
    # ---------------------- Three layer sigmoid ----------------------
    three_layer_sigmoid_test_acc, three_layer_sigmoid_iterations = Early_Stop_Train(
            three_layer_sigmoid, X_train, y_train, X_test, y_test, 
            max_iterations, converge_iterations, "./models/cv/three_layer_sigmoid_%d.h5" % i)
    performance["three_layer_sigmoid"].append(three_layer_sigmoid_test_acc)
    iterations["three_layer_sigmoid"].append(three_layer_sigmoid_iterations)
    
    # ---------------------- Relu Activation ----------------------
    relu_activation_test_acc, relu_activation_iterations = Early_Stop_Train(
            relu_activation, X_train, y_train, X_test, y_test, 
            max_iterations, converge_iterations, "./models/cv/relu_activation_%d.h5" % i)
    performance["relu_activation"].append(relu_activation_test_acc)
    iterations["relu_activation"].append(relu_activation_iterations)
    
    # ---------------------- RMSProp Optimizer ----------------------
    RMSProp_optimizer_test_acc, RMSProp_optimizer_iterations = Early_Stop_Train(
            RMSProp_optimizer, X_train, y_train, X_test, y_test, 
            max_iterations, converge_iterations, "./models/cv/RMSProp_optimizer_%d.h5" % i)
    performance["RMSProp_optimizer"].append(RMSProp_optimizer_test_acc)
    iterations["RMSProp_optimizer"].append(RMSProp_optimizer_iterations)
    
    # ---------------------- Deep Relu Activation ----------------------
    # Bigger model more iterations
    deep_relu_activation_test_acc, deep_relu_activation_iterations = Early_Stop_Train(
            deep_relu_activation, X_train, y_train, X_test, y_test, 
            max_iterations*2, converge_iterations*2, "./models/cv/deep_relu_activation_%d.h5" % i)
    performance["deep_relu_activation"].append(deep_relu_activation_test_acc)
    iterations["deep_relu_activation"].append(deep_relu_activation_iterations)
    
    i += 1


# Quick look at performance
for m in performance:
    print("CV Estimated Class Rate(%s) = %.2f%%" % (m, sum(performance[m])*100 / len(performance[m])))
for m in iterations:
    print("Average Number of Training Iterations(%s) = %.2f" % (m, sum(iterations[m]) / len(iterations[m])))

formatted_names = {
        "one_layer_sigmoid": "Single Hidden layer Sigmoid Activations",
        "two_layer_sigmoid": "Two Hidden layer Sigmoid Activations",
        "three_layer_sigmoid": "Three Hidden layer Sigmoid Activations",
        "relu_activation": "Single Hidden layer ReLU Activations",
        "RMSProp_optimizer": "Single Hidden layer Root MSE Propagation Optimizer",
        "deep_relu_activation": "Deep Network ReLU Activations"
        }

w = 0.1
x_val = -0.2
fold_list = list(range(1, n_folds+1))
for m in performance:
    plt.plot(fold_list, performance[m], label = formatted_names[m])
plt.title('Cross Validation Estimated Classification Success Rates')
plt.xlabel('Fold')
plt.ylabel('Estimate Classification Success Rate (%)')
plt.legend()
plt.show()

plt.clf()
for m in iterations:
    plt.plot(fold_list, iterations[m], label = formatted_names[m])
plt.title('Cross Validation Model Training Iterations')
plt.xlabel('Fold')
plt.ylabel('Number of Training Iterations')
plt.legend()
plt.show()