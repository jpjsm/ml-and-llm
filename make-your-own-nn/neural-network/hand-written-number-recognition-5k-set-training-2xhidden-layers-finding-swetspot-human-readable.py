import csv
import json
import pickle


with open("hand-written-number-recognition-5k-set-training-2xhidden-layers-finding-swetspot-test-results.pickle", "rb") as intput_file:
    test_results = pickle.load(intput_file)
    
accuracy_raw_values = test_results["accuracy_raw_values"]    
field_names = ["hidden_nodes", "learning_rate", "training_epochs", "accuracy", "training_delta_secs", "testing_delta_secs"]    
with open("hand-written-number-recognition-5k-set-training-2xhidden-layers-finding-swetspot-test-results.csv","w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)

    writer.writeheader()
    for accuracy_raw_value in accuracy_raw_values:
        writer.writerow(accuracy_raw_value)

with open("hand-written-number-recognition-5k-set-training-2xhidden-layers-finding-swetspot-test-results.json", "w") as output_file:
    json.dump(test_results, output_file, default=lambda o: o.__dict__, indent=4)
