# Resource - the original BERT paper : https://arxiv.org/abs/1810.04805

# This code should only be run if you have access to a CUDA-enabled GPU or a
# TPU.  It is recommended that you use the tf_bert environment, which
# will save you needing to manually install the correct packages.  This can be
# provided on request.

# This code has been adapted from the excellent tutorial provided here :
# https://www.datacamp.com/tutorial/tutorial-natural-language-processing
# Important note - if you follow the tutorial at the link above, it is
# recommended you follow the accompanying CoLab notebook, as some things are 
# missing from the webpage tutorial; the notebook is linked in the webpage, but
# can be found here : 
# https://tinyurl.com/4hc5mtt6

# import the os and shutil libraries for some file operations
# (we'll use these to automatically tidy up the downloaded dataset)
import os
import shutil

# We'll use TensorFlow to fine-tune our BERT model here
import tensorflow as tf

# import TensorFlow Hub, which is a repository of trained machine learning
# models.  Import it under the alias 'hub'.
import tensorflow_hub as hub

# import TensorFlow Text, which is a library that includes low-level access to
# NLP models and utilities.  KerasNLP is an alternative that allows for higher
# level (more abstracted) utilisation if lower level access is not needed.
import tensorflow_text as text

from official.nlp import optimization # to create AdamW optimiser

# import matplotlib so we can plot learning performance later
import matplotlib.pyplot as plt

# Set up the TensorFlow logger to log at the "ERROR" level; see
# https://www.tensorflow.org/api_docs/python/tf/get_logger for more information
tf.get_logger().setLevel('ERROR')

# Download, extract and tidy the Large Movie Review Dataset
# (This will take a little while the first time it's run, but then won't
# need to download again)
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

# Specify directory of dataset by setting the path to the folder
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

# Specify directory of training data by setting the path to the subfolder
# (basically add the 'train' folder to the dataset path constructed above)
train_dir = os.path.join(dataset_dir, 'train')

# Remove unused "unsup" folder - this is the one that's used for unsupervised
# learning, which we don't need, but if we leave it here it'll throw off the
# learning.
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# Set up an autotune algorithm so that the prefetching of data used during
# training is automatically optimised to your CPU performance
AUTOTUNE = tf.data.AUTOTUNE

# Set batch size and random seed for training.  Remember, batch size is the
# number of training examples shunted through the network in one go, and we use
# a random seed when we carve up the training and validation datasets to
# ensure we don't pick the same examples to be in both (this would be like 
# having the answer to an exam, using that answer, and then marking yourself
# as having passed because your answer matches the answer.ss)
batch_size = 32
seed = 42

# Specify the training data (with 20% carved out for a validation set), giving
# the location, batch size (specified above), proportion of the data to carve
# out for the validation set, which subset we want this to represent (the 
# training data here), and the random seed (specified above) which we fix so
# we don't accidentally take the same data randomly for both training and 
# validation
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batch_size,
    validation_split = 0.2,
    subset = 'training',
    seed = seed)

# Store the class names identified from the training data
class_names = raw_train_ds.class_names

# Set up a prefetch (using the AUTOTUNE specified above) so that as the model
# is training on one batch of data, the prefetch is getting the next batch of
# data ready
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Set up the validation set much as we did with the training set above, but
# this time we tell it it's the validation set we're specifying (so it knows
# to use that 20% we're carving off for this).  The same 20% is carved off each
# time because we're using a random seed to fix this (so it isn't random)
val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batch_size,
    validation_split = 0.2,
    subset = 'validation',
    seed = seed)

# Set up a prefetch on the validation too, just as we did with the training
# data above
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Set up the test set - we don't need to carve anything off this, so we don't
# need to specify a split, or which subset to which it refers, or have a
# random seed (as we're using the whole thing)
test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size = batch_size)

# Set up a prefetch on the test data, as we have with the training and
# validation data
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Now we need to select a BERT model (encoder) to use.  Here we'll use the
# bert_en_uncased_L-12_H-768_A-12 model, which has been pre-trained in English
# using the Wikipedia and BooksCorpus.  The "uncased" bit refers to the fact
# that the text has been translated into lower case before tokenisation, and
# any accents above letters etc have been stripped.  This is a version of the
# model hosted on TensorFlow Hub, which has 12 hidden layers (L-12), 768 hidden
# units (H-768) and 12 Attention Heads (A-12).  You can read more about this
# BERT model at the link specified.
tfhub_handle_encoder = (
    'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')

# We'll also select a pre-processing model which will convert the text into 
# numeric data and put it into tensors (the format needed for a neural 
# network).  Here, we will use the bert_en_uncased_preprocess model - but 
# different BERT models need different pre-processing models.  You can find 
# information about this in the documentation for the model; take a look at the
# link for the BERT model (the encoder) above, and you'll see that there's a
# link to the preprocessing model we're using under "Advanced topics"
tfhub_handle_preprocess = (
    'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')

# Let's load in the preprocessing model we selected above as a Keras Layer,
# and try it out on some sample text to see what it does
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

# Some test text on which we'll try the preprocessing model
text_test = ['this is such an amazing movie!']

# Apply the preprocessing model to the test text
text_preprocessed = bert_preprocess_model(text_test)
    
# Let's have a look at how the data has been preprocessed for this test text.
# For the three preprocessing outputs, we'll only look at the first 12 tokens
# (there are 128 tokens generated for each piece of text in this model, with
# longer text truncated and shorted text padded out with "dummy" tokens of
# value 0 - remember, everything needs to be the same size and shape
# as it goes through a Neural Network)
print (f'Keys       : {list(text_preprocessed.keys())}')
print (f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print (f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print (f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print (f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

# We can see from the above that there are three outputs from the
# preprocessing model :
# 1. the ids of the tokens (these are the numerical
# representations of the tokens, but don't just include the words - they also
# include punctuation as well as tokens for the start and end of the sequence;
# this is why there are 9 word ids for the six word sentence above.  It should
# also be noted that sometimes words will be split into multiple tokens, such
# as those representing different morphemes etc, so that unseen words aren't
# just always treated as completely unknown; see here : 
# https://tinyurl.com/ydhejppn
# 2. the input mask - this has a value of 1 for all positions where there is
# an input token, and a value of 0 for padding tokens (the dummy tokens used
# when the text is shorter than the fixed size - 128 tokens in this case)
# 3. the type ids - this indicates either a 0 or a 1 for each token, depending
# on the "segment" to which the token belongs.  This isn't used here (you'll
# just see 0s) but can be used for identifying, for example, whether the token
# belongs to a question or its corresponding answer (see :
# https://huggingface.co/transformers/v3.2.0/glossary.html).  You'll see that
# used in the bert_nsp.py file, where we look at Next Sentence Prediction.
# 
# The 'keys' print output tells us that these are the three outputs, stored in 
# a dictionary (text_preprocessed).  The 'shape' print output gives us the
# shape of the dictionary - in this case (1, 128), because we have one text
# input here (the sentence we fed in above), and a fixed size of 128 tokens
# for each (or for just the one here).  You can see this if you were to add
# another piece of text to the text_test list - the shape would be (2, 128).

# We'll now write a function that will build our classifier model.  The way
# in which we do this can differ according to the model, so you should consult
# the documentation for it and see the "Basic Usage" section (remember, the
# documentation is available at the URL for the BERT model; that's the one we
# set up as the encoder above)
# We're going to do this in TensorFlow, and we're going to be using the Keras 
# Functional API, rather than the Keras Sequential API you've been used to using
# so far when using TensorFlow.  The Functional API can create models that are 
# more flexible, including being non-linear, or which have multiple inputs and 
# outputs.  The Sequential API that you've used up to this point stacks a linear
# series of layers one on top of the other.
# See : https://www.tensorflow.org/guide/keras/functional for more info.

# We'll define a function that will build a classifier model
def build_classifier_model():
    # Set up an input layer that will take in the raw text (before it's
    # preprocessed)
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    
    # Set up a preprocessing layer that uses the BERT preprocessing model we
    # specified earlier
    preprocessing_layer = hub.KerasLayer(
        tfhub_handle_preprocess, name='preprocessing')
    
    # Apply the preproccessing layer to the raw text inputs we read in to
    # generate the inputs for the BERT model (the encoder)
    # Here we use something known as a "layer call" - this tells TensorFlow
    # that we want the outputs of the layer we put in the brackets to feed into
    # the layer outside of the brackets, and the outputs of this latter layer
    # will be referred to as the name we give before the =.
    # This is a bit like drawing an arrow from one layer to the next, linking 
    # them.  Here, we say that we want the outputs of the input layer (that 
    # takes in the non-preprocessed raw text data) to feed directly into the 
    # preprocessing layer we just set up (the one that's going to do the 
    # preprocessing).  The outputs of the preprocessing layer we have named 
    # 'encoder_inputs', because these will, in turn, feed into the BERT model 
    # (the encoder).
    # You can read more about "layer calls" at the functional API link above.
    encoder_inputs = preprocessing_layer(text_input)
    
    # Set up a layer that will use the BERT encoder model we specified earlier.
    # We also specify that the layer is trainable and so can adapt as the model
    # learns (we are using a pre-trained BERT model, but we can still train
    # it further - this is known as "fine-tuning")
    encoder = hub.KerasLayer(
        tfhub_handle_encoder, trainable=True, name="BERT_encoder")
    
    # Now, just as we did above, we use a "layer call" to link layers.  Here,
    # we specify that we want the encoder_inputs (which, remember, are the
    # outputs of the preprocessing layer, as we specified above) to feed in as
    # the inputs to the encoder layer.  And the outputs from THAT we refer to
    # as 'outputs'.  Remember, the encoder (the BERT model) outputs features of
    # the text, such as the self-attention weights.
    outputs = encoder(encoder_inputs)
    
    # Now we'll add on some additional layers, which we'll build up and store
    # in 'net'.  First, we will specify a layer that will take the 'pooled
    # outputs' of the outputs generated by the encoder layer above.  Pooled
    # outputs have a single representation for each input sequence (ie piece
    # of text), whereas 'sequence outputs' have a representation for each input
    # token (ie each word, punctuation, start / end sequence tokens, dummy 
    # tokens etc).  The BERT model we're using has a 768-dimensional
    # representation for each sequence (if using pooled output) or token
    # (if using sequence output).  Pooled Output is more commonly used when
    # we're trying to classify text (as we are here) or to measure the
    # similarity between sequences (ie how similar are two pieces of text).
    # There's some quite nice explanations here : 
    # https://www.kaggle.com/questions-and-answers/86510
    # We start building up our network by first getting the pooled outputs from
    # the encoder (which itself has had the preprocessing outputs fed into it,
    # and THAT had the raw text fed into it).  So this first bit represents
    # RAW TEXT -> PREPROCESSING MODEL -> PREPROCESSED TEXT -> ENCODER MODEL ->
    # ENCODER OUTPUTS -> POOLED ENCODER OUTPUTS
    net = outputs['pooled_output']

    # Next, we'll link the pooled outputs layer we specified above to a dropout
    # layer with 10% dropout.  Remember, dropout randomly 'switches off' a %
    # of the neurons in our network (by changing the weights coming out of those
    # neurons to 0) in each epoch during training to try to prevent overfitting.
    # By using dropout, we are trying to avoid the model learning to become too
    # reliant on one area of the model as it trains.  Here, our dropout layer 
    # will switch off 10% of the neurons in the pooled output layer in each
    # epoch during training.  We add this to our network, so now we have our
    # pooled output layer that then feeds into a dropout layer, which will
    # randomly switch off some of those outputs during training.
    net = tf.keras.layers.Dropout(0.1)(net)

    # We'll add one final layer onto the network.  This will be a single node 
    # that will represent the final output of the model - the classification.  
    # (Remember, BERT models don't have a "decoder", only an "encoder", so we
    # have to build on the stuff to make the BERT outputs usable for our 
    # purpose - here, that's using it as the basis for training a Sentiment
    # Analysis Classifier Model for movie reviews). The layer is densely 
    # connected (so all the nodes from the previous layer link into it).  We are
    # not using an activation function here, so the raw value will be our 
    # output. To use this for a classification, we'll need to feed it through a 
    # sigmoid function (which will give us the probability of the text being 
    # classified as one of the classes), but we'll do that separately a little 
    # later.
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    
    # Finally, we return our assembled model.
    # When we build a model using the functional API (as opposed to the 
    # sequential API), we start from an input (here our raw text input) and
    # then chain together layers (the network of layers we defined above) which
    # is defined as the model's output.  See here for more info :
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model
    # This is why we return the model in the format Model(input, output)
    return tf.keras.Model(text_input, net)

# Let's now use the function we've written above to build a classifier model
classifier_model = build_classifier_model()

# We can see a plot of the model if we use the following command in the
# iPython console (remove the comment symbol).  Note - you will need to first :
# pip install pydot
# install graphviz : https://graphviz.gitlab.io/download/
# Command :
# tf.keras.utils.plot_model(classifier_model)

# Now we've got our model built, we need to train it.  Obviously the BERT model
# has already been trained, but we can add to this for our application area 
# (fine-tuning), and we'll do this using the IMDB data we downloaded (the same 
# data we used in the Sentiment Analysis training), which has positive and
# negative movie reviews, with each labelled as positive or negative.

# First, we'll specify our loss function and the metrics we want to capture.

# We'll use a BinaryCrossentropy loss function (remember from the training
# that we use such a loss function when we have a classification network where
# we have two possible classifications and we're generating a probability that
# an example belongs to one of them).  Also remember that from_logits=True
# just means normalise the outputs so they fall between 0 and 1 (and therefore
# can be interpreted as a probability).
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# We'll also specify the metrics we want to capture to assess model performance.
# Here, we'll use BinaryAccuracy - this is identical to Accuracy (which we've
# used before, and basically give us the proportion of times the predicted
# classification matched the true classification), except BinaryAccuracy also 
# allows us to optionally specify the classification threshold (the threshold 
# above which the example is classified as class 1, and below which class 0; 
# default is 0.5).  We won't do this here at the moment, but it gives us the 
# option. See here for more details about the various accuracy metrics you can 
# use in Keras : https://keras.io/api/metrics/accuracy_metrics/
metrics = tf.metrics.BinaryAccuracy()

# We'll specify that we want 5 training epochs (remember an epoch is a full
# pass through of all of the training data)
epochs = 5 # default = 5

# We're going to specify a learning rate here, and we're going to use the same
# schedule as was used for the initial BERT training outlined in the original
# BERT paper.  Don't worry about the details of this too much, but basically
# we will use something known as a 'linear decay of a notional initial learning
# rate' with a linear warm-up period over the first 10% of training steps.
# Basically this means that it warms up to a faster learning rate initially,
# then gradually slows down its learning as it fine tunes.
# Because we need to do all this, and specify a warm up period, we need to 
# calculate the total number of training steps, so we can also work out how many
# steps 10% of them (the warm up period) will be.

# First we calcualte the number of steps per epoch, which we can find from the
# cardinality of the dataset (the training data).  This basically just returns
# the batched up size of the dataset (remember the data is split into batches).
# .numpy() converts the output into a numpy array (as the output will be a
# Tensor here.
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()

# Once we know the number of steps per epoch, we can easily calculate the total
# number of steps in the training, as steps per epoch * number of epochs
num_train_steps = steps_per_epoch * epochs

# And we can calculate how many warm up steps we need by just multiplying that
# number by 0.1 (in this case).  We also cast the result as an integer in case
# it's not a whole number (as we can't have a fraction of a training step)
num_warmup_steps = int(0.1 * num_train_steps)

# We now specify the initial learning rate that we want (that will change during
# the warm up period and then subsequently through linear decay).  We use a
# smaller learning rate for this 'fine tuning' training than the initial BERT
# model training; here we use a value of 3e-5 (0.00003) in line with the BERT
# paper.
init_lr = 3e-5

# We'll now set up the optimiser with the information we've specified above.
# Remember, an optimiser is the "correction" algorithm of a neural network that
# decides how to adjust each weight (both in terms of magnitude and direction)
# based on the loss that has been backpropogated (blame assigned) through the
# network.  Also remember that Adam ("Adaptive Moments") is now the most 
# commonly used optimiser, and "if in doubt, use Adam".  So we're going to use 
# Adam here (also the original BERT model was trained with Adam).
# We will create the optimiser, specifying the initial learning rate, total
# number of training steps and number of warmup steps that we set above.
# Just don't forget, you need to use the US spelling, so don't forget your z...
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

# We'll now compile the classifier model we created earlier (when we called the
# build_classifier_model() function that we wrote) using the optimiser, loss
# and metrics that we've specified above.
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

# Now it's time to train the model!  Remember, model training is known as
# 'fitting' (because we're trying to fit a model to the data).  We specify our
# training set, our validation set and the number of epochs (that we set
# earlier).  We'll store the results in a object named history so we can
# access information about the training over time later.
print (f"Training model with {tfhub_handle_encoder}")

history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs)

# Let's see how well our trained model performs on the test set.  We'll look at
# both the loss (error - which we want to be low) and accuracy (which we want
# to be high)
loss, accuracy = classifier_model.evaluate(test_ds)

print (f"Loss: {loss}")
print (f"Accuracy: {accuracy}")

# As we stored the training history, we can plot how loss and accuracy changed
# over time during training for both the training set and the validation set.
# First, we can grab out the history dictionary from the history object.  We'll
# print the keys of the dictionary so we can see what was stored - basically
# accuracy and loss for both the training and validation data
history_dict = history.history
print (history_dict.keys())

# Let's grab out the values we need to plot from the dictionary
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# Now let's plot accuracy and loss for both training and validation.  We'll have
# two subplots - one for loss and one for accuracy, with a red line for the
# training performance, and a blue line for the validation performance.
plt_epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
plt.plot(plt_epochs, loss, 'r', label='Training Loss')
plt.plot(plt_epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(plt_epochs, acc, 'r', label='Training Accuracy')
plt.plot(plt_epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# We don't want to retrain the model every time, so we can save it.  This allows
# us to easily load the model back in later when we want to use it.  We use an
# f sring format here to save the name of the model with the number of epochs
# we used (of course, you could make this as intricate as you like!)
classifier_model.save(f'./saved_models/imdb_bert_model_{epochs}_epochs')

# SEE PART 2 : bert_datacamp_tutorial_part_2.py

