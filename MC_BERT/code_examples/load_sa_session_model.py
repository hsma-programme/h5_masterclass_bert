# It is recommended that you use the supplied bert_mc_cpu environment, which
# will save you needing to manually install the correct packages.

import re # library for regular expression functions
import string # library for string functions

# Import tensorflow and elements of keras (the API to access tensorflow) that
# we need
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing\
    import TextVectorization
    
from tensorflow.keras.callbacks import EarlyStopping

loaded_model = tf.keras.models.load_model('model_from_sa_session')

with open("my_review.txt", "r") as f:
    my_review = f.read()
    
predicted_confidence = loaded_model.predict([my_review])

print ("REVIEW:")
print (my_review)

if predicted_confidence > 0.5:
    print (f"Predicted sentiment is POSITIVE ({predicted_confidence[0]})")
else:
    print (f"Predicted sentiment is NEGATIVE ({predicted_confidence[0]}")

