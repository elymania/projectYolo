"""
Written by: Rahmad Sadli
Website : https://machinelearningspace.com

I finally made this program simple and readable
Hopefully, this program will help some beginners like me to understand better object detection.
If you want to redistribute it, just keep the author's name.

In oder to execute this program, you need to install TensorFlow 2.0 and opencv 4.x

For more details about how this program works. I explained well about it, just click the link below:
https://machinelearningspace.com/the-beginners-guide-to-implementing-yolo-v3-in-tensorflow-2-0-part-1/

Credit to:
Ayoosh Kathuria who shared his great work using pytorch, really appreaciated it.
https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
    Input, ZeroPadding2D, LeakyReLU, UpSampling2D

physical_devices = tf.config.experimental.list_physical_devices('CPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_cfg(cfgfile):
    #opening the cfg file in read mode
    with open(cfgfile, 'r') as file:
    #lines will store all the lines in the cfg file
        lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
    #holder is a dict in which we will have block's attributes and their values
    #after each iteration, the holder is gotten empty and ready to read another block
    holder = {}
    #blocks will store every item of lines
    blocks = []
    #This loop is reading attributes block by block
    #Attribute and key are firstly stored in as the key-value pairs
    #in the holder, afterwards it'll be empty waiting for another block
    #the loop goes until every block is read
    for line in lines:
        if line[0] == '[':
            line = 'type=' + line[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()
    blocks.append(holder)
    return blocks


def YOLOv3Net(cfgfile, model_size, num_classes):

    blocks = parse_cfg(cfgfile) #blocks contains all the attributes read from yolov3.cfg

    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0

    inputs = input_image = Input(shape=model_size)
    inputs = inputs / 255.0 #inputs are normalized with the range 0-1

    for i, block in enumerate(blocks[1:]):
        # If it is a convolutional layer
        if (block["type"] == "convolutional"):
        #read the attributes
            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])

            if strides > 1:
                #adjusting the padding whenever stride > 1
                inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)

            inputs = Conv2D(filters,
                            kernel_size,
                            strides=strides,
                            padding='valid' if strides > 1 else 'same',
                            name='conv_' + str(i),
                            use_bias=False if ("batch_normalize" in block) else True)(inputs)

            if "batch_normalize" in block:
                inputs = BatchNormalization(name='bnorm_' + str(i))(inputs)
            if activation == "leaky":
                inputs = LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)

        elif (block["type"] == "upsample"):
            stride = int(block["stride"])
            inputs = UpSampling2D(stride)(inputs)

        # If it is a route layer
        elif (block["type"] == "route"):
            block["layers"] = block["layers"].split(',') #when we have more than one value, we will concatenate these 2 layers
            start = int(block["layers"][0])

            if len(block["layers"]) > 1: #in case there's more than one value
                end = int(block["layers"][1]) - i
                filters = output_filters[i + start] + output_filters[end]  # Index negative :end - index
                inputs = tf.concat([outputs[i + start], outputs[i + end]], axis=-1)
            else:
                filters = output_filters[i + start]
                inputs = outputs[i + start]

        elif block["type"] == "shortcut":
            #shortcut layer has an attribute (from) from which we take the number of the layer
            #that we have to add to the feature map of the previous layer
            from_ = int(block["from"])
            inputs = outputs[i - 1] + outputs[i + from_]

        # Yolo detection layer
        elif block["type"] == "yolo":
            #from the cfg file we are taking the values and assigning to
            #mask and anchors respectively
            mask = block["mask"].split(",") #there is more than one value
            mask = [int(x) for x in mask] #casting mask values to int
            anchors = block["anchors"].split(",") #there is more than one value
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)] #values are assigned pair-wise
            anchors = [anchors[i] for i in mask]

            n_anchors = len(anchors)

            out_shape = inputs.get_shape().as_list() ###mi sono fermato qui
            #reshaping the input as [None, B*grid size*grid_size, 5+c
            #B is number of anchors, C is number of classes
            inputs = tf.reshape(inputs, [-1, n_anchors * out_shape[1] * out_shape[2], \
										 5 + num_classes])
            #accessing boxes attributes
            box_centers = inputs[:, :, 0:2]
            box_shapes = inputs[:, :, 2:4]
            confidence = inputs[:, :, 4:5]
            classes = inputs[:, :, 5:num_classes + 5]
            #centers, confidence and classes are normalized within the range 0-1
            box_centers = tf.sigmoid(box_centers)
            confidence = tf.sigmoid(confidence)
            classes = tf.sigmoid(classes)
            #box shapes are converted
            anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)
            #creating the position of x and y in the grid
            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)

            cx, cy = tf.meshgrid(x, y)
            cx = tf.reshape(cx, (-1, 1))
            cy = tf.reshape(cy, (-1, 1))
            cxy = tf.concat([cx, cy], axis=-1)
            cxy = tf.tile(cxy, [1, n_anchors])
            cxy = tf.reshape(cxy, [1, -1, 2])

            strides = (input_image.shape[1] // out_shape[1], \
                       input_image.shape[2] // out_shape[2])
            box_centers = (box_centers + cxy) * strides

            prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
            #doing 3 prediction across the scale
            if scale:
                out_pred = tf.concat([out_pred, prediction], axis=1)
            else:
                out_pred = prediction
                scale = 1
        #route and shortcut need output feature from previous layers,
        #in each iteration it will be kept track fo feature maps and output filters
        outputs[i] = inputs
        output_filters.append(filters)

    model = Model(input_image, out_pred)
    model.summary()
    return model