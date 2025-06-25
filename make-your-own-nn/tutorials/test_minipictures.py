import numpy
import matplotlib.pyplot as plt
import json

with open("/mini-pictures/minipictures_train.json", 'r') as input_json:
    minipictures_train = json.load(input_json)


pics_by_label = dict()

for pic in minipictures_train:
    if pic["Label"] not in pics_by_label:
        pics_by_label[pic["Label"]] = []

    pics_by_label[pic["Label"]].append(pic["Inputs"])
    
fig = plt.figure(figsize= (3, 3))
i = 1
for key in pics_by_label:
    for image in pics_by_label[key]:
        image_array = numpy.asfarray(image).reshape((3,3))
        ax = fig.add_subplot(3,3,i) 
        ax.imshow(image_array, cmap='Greys', interpolation='None')
        # ax.title(f"Label '{key}'")
        #ax.show()
        i += 1

