# BinaryImageClassificationSuite.py
# Used for the creation and implementation of a custom binary image classifier.

# Cole Lightfoot - 9 May 2021 - https://github.com/cole8888/

# Imports, Tensorflow imports are later
import os
import sys
import csv
import time
import timeit
import argparse
import progressbar
import numpy as np
import PIL.Image as Image

from os.path import join as joinPath
from multiprocessing import cpu_count
from threading import Timer,Thread,Event

# Command line arguments.
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--home", 
    type=str,
    default=None,
    help="Directory where models, labels and other files are saved and loaded from.")
parser.add_argument(
    "--epochs",
    type=int,
    default=4,
    help="Number of epochs to train for"
)
parser.add_argument(
    "--finetune_epochs",
    type=int,
    default=4,
    help="Number of epochs to do finetuning on after initial training."
)
parser.add_argument(
    "--mode",
    type=str,
    default=None,
    help='Select "train", "predict" or "clean" mode.',
)
parser.add_argument(
    "--mem",
    type=int,
    default=6,
    help='Number of gigabytes of system memory you would like to dedicate to preprocessing. (This is a rough estimate)',
)
parser.add_argument(
    "--images",
    type=str,
    default=None,
    help='Directory of images you want to predict on. When using "clean" mode this is the root directory to recusively scan from. Only valid in "predict" and "clean" modes.',
)
parser.add_argument(
    "--class1_thresh",
    type=int,
    default=10,
    help='Threshold a prediction needs to exceed in order to not be placed into UNSURE for this class. Only needed during prediction.',
)
parser.add_argument(
    "--class2_thresh",
    type=int,
    default=5,
    help='Threshold a prediction needs to exceed in order to not be placed into UNSURE for this class. Only needed during prediction.',
)
parser.add_argument(
    "--pthreads",
    type=int,
    default=cpu_count(),
    help='Number of threads to use for image preprocessing. Leave blank to automatically use the number of threads on your CPU.',
)
parser.add_argument(
    "--tworkers",
    type=int,
    default=4,
    help='Number of worker threads to use for training. Only valid during training.',
)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    help="Dataset you are using."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Number of images in each batch, reduce this if memory errors occur."
)
parser.add_argument(
    "--chunk_size",
    type=int,
    default=1400,
    help='Number of images per chunk per GB. Probably want to keep this at 1400',
)
parser.add_argument(
    "--size",
    type=int,
    default=224,
    help="Size of cropped input image to the network, make sure this is right for the selected model.",
)
parser.add_argument(
    "--val_split",
    type=float,
    default=0.04,
    help="Size of the split of the training dataset to use for validation images.",
)
parser.add_argument(
    "--base_lr",
    type=float,
    default=0.0001,
    help="Base learning rate value.",
)
parser.add_argument(
    "--fine_layer_start",
    type=int,
    default=5,
    help="From which layer onward of the model should we start finetuning?",
)
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="How many steps to do per epoch of training? Leave balnk for auto.",
)
parser.add_argument(
    "--val_steps",
    type=int,
    default=None,
    help="How many steps of validation to do per epoch of training? Leave blank for auto.",
)
parser.add_argument('--no_sort', action="store_true", help='Do not move the images when sorting. This is mostly for testing purposes.')
parser.add_argument('--eval', action="store_true", help='Perform a final evaluation of the model when training is complete.')
parser.add_argument('--finetune', action="store_true", help='Do finetune training.')
parser.add_argument('--chunks', action="store_true", help='Handle the set of images to predict on in chunks, good when not enough memory is available.')
parser.add_argument('--no_warn', action="store_true", help='Disable my built-in warnings. Only do this if you know what you are doing. -_-')
parser.add_argument('--tf_warn', action="store_true", help='Enable tensorflow logging and warnings to the command line.')
parser.add_argument('--print_detect', action="store_true", help='Print out the detections as they are made.')
parser.add_argument('--print_prep', action="store_true", help='Print out the status of the image preprocessor.')
parser.add_argument('--csv', action="store_true", help='Save a CSV file to the home directory with prediction results.')
parser.add_argument('--save_score', action="store_true", help='Append the confidence to the detected file\'s name.')
args = parser.parse_args()

# Catch issues in arguments. More are handled later in the program.
if(args.size < 1 and args.mode != "clean"):
    raise ValueError("Image size must be greater than 1.")
elif((args.home is None) and args.mode != "clean"):
    raise ValueError("You must specify a home folder where models and other files are saved and loaded from.")

# See if we should disable tensorflow warnings and logs. Must be done prior to TF imports
from silence_tensorflow import silence_tensorflow
if(not args.tf_warn):
    silence_tensorflow()

# Do tensorflow imports now that we know if user wants to silence warnings.
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from keras_preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Prepare the preprocessing function for mobilenetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Used in preprocessing timing.
t = time.time()

# List of images used in preprocessing
images = []

# List of image names used for identifying them, moving them, etc...
# This does not include the whole filepath, just the file name.
image_names = []

# List of preprocessed images.
# This will contain raw tensor data so don't print any of these out unless you like command line puke
prep_images = []

# List of Corrupt / incompatible image files images found during detection.
baddies = []

# 2D List of results returned by the prediction function. This can be used to make a csv log,
# and or be passed to the sorting function to move all the images to the correct places.
results = []

# 2D list of images to tell the preprocessing threads which ones they need to handle.
# This is used in both single and chunk mode.
image_chunks = []

# Keeps track of how many detections are made for every class across chunks. Only used in chunked mode.
scores_global = []

# List of possible detection classes, populated later
class_list = []

# Global 2D list containing all the images split into chunks. Only used in chunked mode.
chunks_global = []

# Counter to help coordinate the threads, counts how many threads have completed.
# Lets us figure out when the last one is the current thread so that we do not kill it.
t_done = 0

# Counter / index used when chunks are enabled, this lets us choose which images we want and what
# chunk we are on currently. Only used in chunked mode.
current_chunk = 0

# Keep track of the number of images before chunking. Only used in chunked mode.
num_images = 0

# Used for timing the image preprocessing process.
start_time = 0

# Training images directory, defined later.
TRAIN_DIR = ""

# A flag to help handle when we have more image processing threads than images.
not_enough_images = False

# Model we are training or using to predict, defined later.
model = None

# Flag to see if we have finished preprocessing images for prediction.
done_preproc = False

# Flag to see if we have finished loading the model, used only in chunked mode.
done_loading_model = False

# Run the training process, save the models and labels.
def train():
    IMG_SIZE = (args.size, args.size)

    # Prepare the datasets
    train_dataset = image_dataset_from_directory(TRAIN_DIR,
                                                validation_split=args.val_split,
                                                subset="training",
                                                shuffle=True,
                                                seed=123,
                                                batch_size=args.batch_size,
                                                image_size=IMG_SIZE,
                                                label_mode='binary'
    )

    validation_dataset = image_dataset_from_directory(TRAIN_DIR,
                                                    validation_split=args.val_split,
                                                    subset="validation",
                                                    seed=123,
                                                    shuffle=True,
                                                    batch_size=args.batch_size,
                                                    image_size=IMG_SIZE,
                                                    label_mode='binary'
    )

    class_names = train_dataset.class_names

    print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    
    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)

    # For initial training, leave this off.
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(args.size, args.size, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    # Initialize the optimizer
    opt = tf.keras.optimizers.Adam(lr=args.base_lr)

    # Allow optimizer to use tensorcores
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    # Do the initial training round
    history = model.fit(train_dataset,
                        epochs=args.epochs,
                        workers=args.tworkers,
                        validation_data=validation_dataset,
                        validation_steps=args.val_steps,
                        use_multiprocessing=True,
                        steps_per_epoch=args.steps)
                        # steps_per_epoch=100)


    # See if we need to run finetune training also.
    if(args.finetune):
        print("Starting finetuning...")
        base_model.trainable = True

        # Freeze all the layers before the `args.fine_layer_start` layer
        for layer in base_model.layers[:args.fine_layer_start]:
            layer.trainable =  False

        # Initialize the optimizer
        opt2 = tf.keras.optimizers.RMSprop(lr=args.base_lr/10)

        # Allow optimizer to use tensorcores
        opt2 = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt2) 
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    optimizer = opt2,
                    metrics=['accuracy'])

        # - 1 because there were some extra epochs for some reason.
        total_epochs = args.epochs + args.finetune_epochs - 1

        # Do the finetuning training round
        history_fine = model.fit(train_dataset,
                            epochs=total_epochs,
                            steps_per_epoch=args.steps,
                            # steps_per_epoch=100,
                            initial_epoch=history.epoch[args.epochs-1],
                            workers=args.tworkers,
                            validation_data=validation_dataset,
                            validation_steps=args.val_steps,
                            use_multiprocessing=True)

    # Save the model
    tf.saved_model.save(model, args.home)

    # Evaluate the model if wanted
    if(args.eval):
        final_loss, final_accuracy = model.evaluate(
            validation_dataset,
            steps = int(tf.data.experimental.cardinality(validation_dataset)),
            workers = args.tworkers)
        print("Final loss: {:.2f}".format(final_loss))
        print("Final accuracy: {:.2f}%".format(final_accuracy * 100))

    # Save the class labels for prediction later on.
    labels = ','.join(sorted(class_names))
    with open(os.path.join(args.home, 'labels.txt'), 'w') as f:
        f.write(labels)


# --- All other functions are for prediction ---

# Parse the 2D list of prediction results.
# Used to create directory structure and move images to their proper directories
# as well as append the accuracy to the begining of the filename and if the item is AIO append the
# top category to the filename. Used only during prediction.
def sort_set(data):
    img_dir = args.images
    sort_dir = os.path.join(img_dir, 'SORTED')

    # Create directory structure
    if(not os.path.exists(sort_dir)):
        os.mkdir(sort_dir)
        os.mkdir(os.path.join(sort_dir, 'UNSURE'))
    else:
        i=2
        while os.path.exists(os.path.join(img_dir, 'SORTED' + str(i))):
            i+=1
        sort_dir = os.path.join(img_dir, 'SORTED' + str(i))
        os.mkdir(sort_dir)
        os.mkdir(os.path.join(sort_dir, 'UNSURE'))
    # Extract classes from the data and create a new directory, then create a subdirectory in UNSURE to put any images from this category which do not meet the threshold.
    # (Helps sort mislabbeled images later)
    for item in class_list:
        os.mkdir(os.path.join(sort_dir, str(item)))
        os.mkdir(os.path.join(sort_dir, "UNSURE", str(item)))
    # Move the images to the specified location.
    # Source and destination must be on the same partition!
    for i in data:
        # This fails sometimes for unknown IO reasons, catch exceptions and move on.
        try:
            # Get the right threshold.
            thresh = args.class2_thresh
            if(i[1] == str(class_list[0])):
                thresh = args.class1_thresh
            
            # Hold file extenstions here.
            file_ext = ""
            # handle the loss of file extentions due to truncating the filename later. Longest possible is JPEG, 4 chars.
            if(len(i[0]) >= 130):
                file_ext = str(i[0])[-4:]
            
            # [:100] is to ensure that the files do not exceed the maximum file name or path length on most file systems.
            # If we want to save scores to the filename:
            if(args.save_score):
                # See if detection confidence meets threshold.
                if(float(i[2]) < thresh):
                    os.rename(os.path.join(img_dir, str(i[0])), os.path.join(sort_dir, "UNSURE", str(i[1]), str(i[2]) + '_' + str(i[0])[:130] + file_ext))
                else:
                    os.rename(os.path.join(img_dir, str(i[0])), os.path.join(sort_dir, str(i[1]), str(i[2]) + '_' + str(i[0])[:130] + file_ext))
            # If we do not want to save scores to the filename:
            else:
                # See if detection confidence meets threshold.
                if(float(i[2]) < thresh):
                    os.rename(os.path.join(img_dir, str(i[0])), os.path.join(sort_dir, "UNSURE", str(i[1]), str(i[0])[:130] + file_ext))
                else:
                    os.rename(os.path.join(img_dir, str(i[0])), os.path.join(sort_dir, str(i[1]), str(i[0])[:130] + file_ext))
        except:
            baddies.append(os.path.join(args.images, str(i[0])))

# Load an image and do preprocessing on it to prepare it for prediction.
def load_image(filename):
    # Load the image
    img = load_img(filename, target_size=(args.size, args.size))
    # Convert to array
    img = img_to_array(img)
    # Reshape into a single sample with 3 channels
    img = img.reshape(1, args.size, args.size, 3)
    # Center pixel data
    img = img.astype('float32')
    return img

# Preproccess images. Each thread will execute it's own copy of this function.
# Arguments is the image array and the thread number.
def preproc(images, tnum):
    # List of pre-proccessed images
    global prep_images

    # List of image names
    global image_names

    # List of Corrupt / incompatible files
    global baddies

    # Get all files and join them to their paths then pre-proccess images and add them to the list. Ignore bad files.
    if(args.print_prep):
        print("Started pre-processing " + str(len(images)) + " images on thread " + str(tnum))
    for img in images:
        # Load and prepare the image. If it fails, skip it and move on.
        try:
            img_path = os.path.join(args.images, img)
            
            # Actually do the preprocessing.
            image_data = load_image(img_path)

            image_names.append(img)
            prep_images.append(image_data)
        except:
            baddies.append(os.path.join(args.images, img))
    if(args.print_prep):
        print("Finished Image pre-processing for thread " + str(tnum))

    # See if last thread, also determined if this thread should be killed.
    isLast()

# Restore a pretrained custom model 
def restore_model():
    global model
    global done_preproc
    global done_loading_model

    # Load the model we trained previously.
    print("\nLoading pretrained model...")
    model = tf.keras.models.load_model(args.home)
    
    # Not sure why but the model is not immediately ready until you try to predict. Do this now to fix issues with the progress bar.
    model.predict(prep_images[0])
    print("\nSuccessfully loaded pretrained model!")

    # If we are in chunked mode indicate that we have loaded the model and kill this thread.
    if(args.chunks):
        done_loading_model = True
        sys.exit()

    # Preprocessing may not have finished yet! Make sure it is done.
    while done_preproc is False:
        time.sleep(0.01)
    
    # Strat the prediction process.
    predict()

# Determine if the thread calling this method is the last one to finish, start prediction if so.
def isLast():
    global t_done
    global prep_images
    global start_time
    global not_enough_images
    global done_preproc

    # Increment done threads counter.
    t_done += 1

    # If last thread, do some things differently before exiting.
    if(t_done == args.pthreads or not_enough_images):
        print("\nPreprocessing " + str(len(prep_images)) + " images took %s seconds" % round((time.time() - start_time), 3))
        done_preproc = True

        if(args.chunks):
            # Loading of model may not have finished yet! Make sure it is done.
            while done_loading_model is False:
                time.sleep(0.01)
            # Start the prediction process.
            predict()
    if(not args.chunks):
        # Kill this particular thread, we do not need it anymore.
        sys.exit()

# Function to split up images into chunks, this is used for both threading and splitting up the list in chunked mode.
# Option thread is for assigning to threads, chunk is for assigning chunks.
def divide_chunks(images, n, e, option, not_enough_images = False):
    num = 0
    chunks = []

    # check if we have less images than threads, handle appropriately
    if(not_enough_images):
        chunks.append([])
        for i in range(0, len(images)):  
            chunks[0].append(images[i])
        return chunks
    
    # looping till n = length of images minus extras
    for i in range(0, len(images)-e, n):  
        chunks.append(images[i:i + n])
        num +=1
    
    # Append the extra images to the first chunk, to an additional one at the end if in chunked mode.
    if(e != 0):
        # If we are running in chunked mode we want the extras to be on a seperate chunk,
        # not on the first chunk like when assigning to threads since we may run out of memory on that chunk.
        if(option == "chunk"):
            if(len(images)%(args.mem * args.chunk_size) > 0):
                chunks.append([])
                for i in range(len(images)-e, len(images), 1):
                    chunks[num].append(images[i])
                    i += 1
        else:
            for i in range(len(images)-e, len(images), 1):
                chunks[0].append(images[i])
                i += 1
    return chunks

# Prepare and start preprocessing threads. Reused with both single and chunked mode.
def preprocRun():
    global prep_images
    global image_chunks
    global start_time
    global not_enough_images
    global images
    global current_chunk
    global chunks_global

    # We need a local version of this data since we may modify it.
    num_threads = args.pthreads

    # Since we reuse this function for both single and chunked mode, see if we need to do chunked things.
    if(args.chunks):
        images = chunks_global[current_chunk]
        print("\n--------------------\nCHUNK " + str(current_chunk+1) + " of " + str(len(chunks_global)))
    
    # Number of images to assign to each thread.
    elements = int(len(images) / args.pthreads)

    # If we have more threads than images, set we need to inform the functions that follow and set the threads to 1
    if(len(images) < args.pthreads):
        not_enough_images = True
        num_threads = 1
    
    # Get 2D list of images for the threads. Each thread will have it's own list to work on.
    image_chunks = divide_chunks(images, elements, len(images) % args.pthreads, "thread", not_enough_images)
    
    # Start the threads
    start_time = time.time()
    for i in range(num_threads):
        Thread(target=preproc, args=(image_chunks[i], i)).start()
    
    # Kill this thread
    sys.exit()

# Function to write data to the CSV file.
def write_csv(data):
    header = ["Filename", "Label", "Confidence"]
    csv_book = open(os.path.join(args.images, "results.csv"), 'w')
    writer = csv.writer(csv_book)
    with csv_book:
        writer.writerow(header)
        writer.writerows(data)
    csv_book.close()

def predict():
    global prep_images
    global current_chunk
    global t_done
    global class_list
    global results
    global image_names
    global num_images
    global model

    # Index to use while navigating the results list (Y axis). Also used to navigate the names list and assign proper names
    index = 0

    # Keeps track of the number of detections per category for this chunk.
    scores = []

    # Populate the scores list
    for i in range(len(class_list)):
        scores.append(0)

    print("\nRunning predictions, this may take a while...\n")

    # Display the progressbar, but only if we are not printing results for each image.
    if(not args.print_detect):
        # Progress bar to show prediction progress.
        widgets = [progressbar.Timer(), '|', progressbar.Percentage(), '|', progressbar.ETA(), ' ', progressbar.Bar()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(image_names), term_width=100).start()

    # Start timer for predictions.
    start = timeit.default_timer()

    # This will hold a 1 if confidence is >=0 or 0 if <0.
    # 0 means class 1 and 1 means class 2.
    class_prediction = 0

    # Run the classifier on all the images, store and or print results.
    for image in prep_images:
        # Run prediction and get confidence.
        out = model.predict(image)
        confidence = (out[0])

        # Determine which class the prediction belongs to
        if(confidence >= 0):
            class_prediction = 1
        else:
            class_prediction = 0

        # Get the class name.        
        class_name = class_list[class_prediction]

        # Increment the score for this class
        scores[class_prediction] += 1

        # Append results to the list.
        results.append([])
        results[index].append(image_names[index])
        results[index].append(str(class_name))
        results[index].append(round(float(abs(out[0])), 3))

        # Print summary for the image, if wanted. If this is enabled we do not want to see the progress bar.
        if(args.print_detect):
            print("\nImage: " + image_names[index])
            print("Predicted class: ", str(class_name))
            print("Confidence:  \t", round(float(abs(out[0])),3))
        else:
            # Increment progressbar
            pbar.update(index)

        # Increment index (move down a row)
        index += 1

    # Finished predicting, stop timer
    stop = timeit.default_timer()

    print("\n\nFinished prediction!")

    # Do not sort the images if no_sort is true
    if(not args.no_sort):
        sort_set(results)

    # See if we need to save the results to a csv
    if(args.csv):
        write_csv(results)

    # Print the class totals.
    print("\nClass totals:")
    for i in range(0, len(scores)):
        print("\t" + str(class_list[i]) + ": " + str(scores[i]) + "\t(" + str(round(100 * (scores[i]/len(prep_images)), 4)) + "%)")

    # List all incompatible images, these should be removed by the user
    # and they will be left behind in the image directory if we are sorting the images.
    if(len(baddies) > 30):
        print("\n" + str(len(baddies)) + " Incompatible Images, too many to show.")
    elif(len(baddies) > 0):
        print("\n" + str(len(baddies)) + " Incompatible Images:")
        for i in baddies:
            print(i)
    
    # Print out statistics
    print('\nSorted ' + str(index) + ' images in ' + str(round(stop - start, 2)) + ' seconds, ' + str(round(index/(stop - start), 2)) + " imgs/sec")

    # if we are in chunked mode, there may be more things to do.
    if(args.chunks):
        for i in range(0, len(class_list)):
            scores_global[i] += scores[i]

        # Evaluate whether we should run another chunk.
        if(not current_chunk+1 < len(chunks_global)):
            print("\n--------------------\nOverall class totals")
            for i in range(0, len(scores_global)):
                print("\t" + str(class_list[i]) + ": " + str(scores_global[i]) + "\t(" + str(round(100 * (scores_global[i]/num_images), 4)) + "%)")
        else:
            # Reset the processed images list, this frees up most of the memory.
            prep_images = []
            # Reset the image names list inbetween chunks.
            image_names = []
            # Reset the threads done variable to ensure we can detect finished threads later.
            t_done = 0
            # Increment to indicate which run we are currently on. This is also used to select the proper images.
            current_chunk += 1
            # Reset the results array.
            results = []

            # Run the next chunk
            preprocRun()

# Function to delete incompatible files which may be scrapped. Helps save drive space.
def cleanFolder():
    i = 0
    exts = [".mp4", ".mkv", ".webm", ".gif", ".mp3", ".m4a", ".ico"]
    for r, d, f in os.walk(args.images):
        for file in f:
            for j in exts:
                if(file.lower().endswith(j)):
                    os.remove(os.path.join(r, file))
                    i+=1
                    break
    print("\nDeleted " + str(i) + " files.")

if(args.mode == "train"):
    # Cover all the bases where issues are likely to arise.
    if(args.dataset is None):
        raise ValueError("You must provide a path to the training dataset when using train mode.")
    elif(args.tworkers < 1):
        raise ValueError("Invalid number of training workers. Must be greater than 1.")
    elif(args.epochs < 1):
        raise ValueError("Invalid number of training epochs. Must be greater than 1.")
    elif(args.batch_size < 1):
        raise ValueError("Invalid training batch size. Must be greater than 1.")
    elif(args.val_split <= 0 or args.val_split >= 1):
        raise ValueError("Invalid validation split ratio. Must be between 0 and 1, non inclusive.")
    elif(args.base_lr <= 0 or args.base_lr >= 1):
        raise ValueError("Invalid base learning rate. Must be between 0 and 1, non inclusive.")
    elif(args.fine_layer_start < 0):
        raise ValueError("Invalid fine tune start layer. Must be greater than 0.")

    # Define the training dataset.
    TRAIN_DIR = os.path.join(args.dataset, "train")
    
    # Check / create home path. Verify user wants to overwrite if it exists already.
    if(not os.path.isdir(os.path.join(args.home))):
        os.makedirs(os.path.join(args.home))
    elif(not args.no_warn):
        if(input(str(os.path.join(args.home)) + " already exists, are you sure you want to overwrite? (y/N): ").lower() != "y"):
            exit()
    train()

elif(args.mode == "predict"):
    # Cover all the bases where issues are likely to arise.
    if(args.images is None):
        raise ValueError("You must pass an image directory path when using prediction mode.")
    elif(args.pthreads < 1):
        raise ValueError("Invalid number of preprocessing threads. Must be greater than zero.")
    elif(args.class1_thresh < 0 or args.class2_thresh < 0):
        raise ValueError("Invalid detection threshold. Must be greater than zero.")

    if(args.pthreads > cpu_count() and not args.no_warn):
        if(input("\nWarning, you are going to use " + str(args.pthreads) + " threads. Your CPU has " + str(cpu_count()) + " threads.\nThis may cause lower performance than if you used " + str(cpu_count()) + " threads.\nDo you want to continue? (y/N): ").lower() != "y"):
            exit()
    
    # Get all compatible files
    for file in os.listdir(args.images):
            if(file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".JPEG") or file.endswith(".PNG")):
                images.append(file)
    if(len(images) < 1):
        raise ValueError("Invalid number of images to predict on. Must be atleast one.")
    
    # Make sure user is aware they may potentially run out of RAM
    if(not args.chunks and len(images) > 10000 and not args.no_warn):
        if(input("\nWarning, you are predicting on " + str(len(images)) + " images without using chunks. Are you sure you want to do this?\nYou may lose the prediction progress if you run out of RAM.\nDo you want to continue? (y/N): ").lower() != "y"):
            exit()

    # Open the classes list file and store a local copy of the classes.
    class_list_file = os.path.join(args.home, "labels.txt")
    try:
        with open(class_list_file) as f:
            for line in f:
                inner_list = [elt.strip() for elt in line.split(',')]
                for i in inner_list:
                    class_list.append(i)
                # Should only be one line, break immediately
                break
    except:
        raise FileNotFoundError("Could not locate or open the labels file at: ", os.path.join(args.home, "labels.txt"))

    print("\nClass list: ", class_list)
    print("Number of pre-processing threads: ", args.pthreads)

    # Start thread to restore the trained model while we do preprocessing to save time.
    Thread(target = restore_model).start()

    # See if chunk mode is enabled
    if(args.chunks):
        if(args.mem < 1):
            raise ValueError("Memory cannot be less than 1GB")
        elif(args.chunk_size < 1):
            raise ValueError("Chunk size must be greater than 1.")

        # Prepare the global scores for all the clases across all chunks.
        for i in class_list:
            scores_global.append(0)

        # 1400 images is roughly equal to 1 GB of ram, this is the default chunk_size
        # If number of images is too small to fit into a chunk, then run single.
        if(len(images) < args.mem * args.chunk_size):
            args.chunks = False
            preprocRun()
        else:
            # Tell the other methods how many images we have in total.
            num_images = len(images)

            # Number of runs we need to do
            runs = int(len(images)/(args.mem * args.chunk_size))

            # Check if we have extras, need a spare run.
            if(len(images)%(args.mem * args.chunk_size) > 0):
                runs += 1
            
            # Find out how many extras we have which will be placed in an additional chunk at the end.
            extras = int(len(images) % int(args.mem * args.chunk_size))

            # Print a summary so user knows wtf is going on.
            print("\nTotal images: " + str(len(images)))
            print("Images per run: " + str(args.mem * args.chunk_size))
            print("Number of runs: " + str(runs))
            print("Images on final run: " + str(extras))

            # Split the set of images into the correct sized chunks.
            chunks_global = divide_chunks(images, args.mem * args.chunk_size, extras, "chunk")
    
    # Start preprocessing the images, this will handle thread creation and starting the predictions.
    preprocRun()
elif(args.mode == "clean"):
    cleanFolder()
else:
    raise ValueError("Invalid mode. Please use \"predict\", \"train\", \"clean\" modes")
