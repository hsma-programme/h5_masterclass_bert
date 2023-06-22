# It is recommended that you use the supplied bert_mc_cpu environment, which
# will save you needing to manually install the correct packages.

import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

# Let's load back in the model we saved in Part 1
# We'll ask the user to specify the number of epochs corresponding to the model
# that they want to load back in
chosen_epochs = int(input("Please specify # of epochs of model to load: "))
loaded_model = tf.saved_model.load(
    f'./saved_models/imdb_bert_model_{chosen_epochs}_epochs')

# Let's set up some pieces of text to test our trained model
examples = [
    'A fantastic movie that really pushed the boundaries of storytelling.',
    'The movie was really good, but I think sometimes it drifted.',
    'The movie was mediocre, there are better films out there..',
    'I thought it was ok, but not what I expected',
    'This is the worst thing I have ever watched!!!'
]

# Now let's use the loaded model to predict the classification (sentiment) of
# these examples.  Remember, we'll need to apply a sigmoid function to the
# outputs as we didn't have an activation function on the final layer of our
# network.  We use tf.constant to turn the list of examples (a list is a tensor-
# like object) into a constant tensor (one that cannot be changed), which is the
# input required to the model to allow it to make predictions.
example_results = tf.sigmoid(loaded_model(tf.constant(examples)))

# Now let's print each example text alongside the probability that the text
# belongs to class 1 ("positive sentiment"), and the predicted classification
# based on a threshold
classification_threshold = 0.5
i = 0
for example in examples:
    print (example)
    print (f"{example_results[i][0]:.6f}")
    if example_results[i][0] > classification_threshold:
        print ("POSITIVE")
    else:
        print ("NEGATIVE")
    print ()
    i += 1

