from NeuralNetworkHidden2 import NeuralNetworkHidden2
import csv
from datetime import datetime, timedelta
import itertools
import json
import numpy
import pickle


# Constant definitions
input_nodes = 784
output_nodes = 10
significant_digits = 100000
train_subset_size = 5000
test_subset_size = 500

# MNIST datasets
with open("../mnist_dataset/mnist_train.csv", "r", encoding="utf-8", newline='\n') as input:
    mnist_train = input.readlines()

with open("../mnist_dataset/mnist_test.csv", "r", encoding="utf-8", newline='\n') as input:
    mnist_test = input.readlines()

# Varying parameters
hidden_nodes_values = [50, 75, 100, 150, 200, 300, 400]

learning_rate_values = [0.05, 0.075, 0.09, 0.1, 0.2, 0.3, 0.5, 0.75]

epochs_values = [1, 2, 3, 5, 8, 13]

# Finding max accuracy
accuracy_values = {}
accuracy_raw_values = []
test_start = datetime.now()
for hidden_nodes, learning_rate, epochs in itertools.product(hidden_nodes_values, learning_rate_values, epochs_values):
    # Neural network definition
    nn = NeuralNetworkHidden2(input_nodes, hidden_nodes,output_nodes,learning_rate)

    # Training Neural Network
    training_data_list = numpy.random.choice(mnist_train, 5000)

    training_start = datetime.now()
    for _ in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            label = all_values[0]
            input_values = (numpy.asfarray(all_values[1:])/255.0*0.99)+.01
            target_values = numpy.zeros(output_nodes) + 0.01
            target_values[int(all_values[0])] = 0.99
            nn.train(input_values, target_values)

    training_delta = datetime.now() - training_start

    # Test Neural Network performance        
    test_data_list = numpy.random.choice(mnist_train, 500)

    scorecard = []
    testing_start = datetime.now()
    for test_case in test_data_list:
        all_values = test_case.split(',')
        label = all_values[0]
        expected_label = int(label)

        outputs = nn.query((numpy.asfarray(all_values[1:])/255.0*0.99)+.01)
        actual_label = numpy.argmax(outputs)
        values = [v[0] for v in outputs]
        
        matches = '!='
        score = 0
        if expected_label == actual_label:
            matches = '=='
            score = 1

        scorecard.append(score)

    testing_delta = datetime.now() - testing_start

    accuracy = sum(scorecard)/len(scorecard)
    accuracy_str = f"{int(significant_digits*accuracy)/significant_digits:8.6f}"
    if accuracy_str not in accuracy_values:
        accuracy_values[accuracy_str] = []

    accuracy_values[accuracy_str].append((hidden_nodes, learning_rate, epochs))

    iteration_values = { "accuracy":accuracy, 
                                 "hidden_nodes":hidden_nodes, 
                                 "learning_rate":learning_rate, 
                                 "training_epochs":epochs,
                                 "training_delta_secs":training_delta.total_seconds(),
                                 "testing_delta_secs":testing_delta.total_seconds()}
    
    print(f"[{datetime.now().isoformat()}]: {iteration_values}")

    accuracy_raw_values.append(iteration_values)

test_duration_secs = (datetime.now() - test_start).total_seconds()

Top10Accuracies = sorted(accuracy_values.keys(),reverse=True)[:10]

for topAccuracy in Top10Accuracies:
    print(f"{topAccuracy} <-- {accuracy_values[topAccuracy]}")

test_results = {"input_nodes":input_nodes,
               "output_nodes":output_nodes,
               "train_subset_size":train_subset_size,
               "test_subset_size":test_subset_size,
               "hidden_nodes_values": hidden_nodes_values,
               "learning_rate_values":learning_rate_values,
               "epochs_values":epochs_values,
               "test_duration_secs":test_duration_secs,
               "accuracy_raw_values":accuracy_raw_values}

with open("hand-written-number-recognition-5k-set-training-2xhidden-layers-finding-swetspot-test-results.pickle", "wb") as output:
    pickle.dump(test_results, output, pickle.HIGHEST_PROTOCOL)
    
field_names = ["hidden_nodes", "learning_rate", "training_epochs", "accuracy", "training_delta_secs", "testing_delta_secs"]    
with open("hand-written-number-recognition-5k-set-training-2xhidden-layers-finding-swetspot-test-results.csv","w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)

    writer.writeheader()
    for accuracy_raw_value in accuracy_raw_values:
        writer.writerow(accuracy_raw_value)

with open("hand-written-number-recognition-5k-set-training-2xhidden-layers-finding-swetspot-test-results.json", "w") as output:
    json.dump(test_results.__dict__, output, default=lambda o: o.__dict__)
