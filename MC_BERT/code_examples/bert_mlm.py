# Code examples taken and adapted from :
# https://huggingface.co/blog/bert-101
#
# Example of BERT's Masked Language Modelling
# 
# The transformers library must be installed for this code :
# pip install transformers
#
# It is recommended that you use the supplied bert_mc_cpu environment, which
# will save you needing to manually install the correct packages.
#
# Important!  The first time you run this it will need to download some large
# files.  Be aware of this if you're running it for the first time,
# particularly if you have a slower internet connection.

# This import provides us with a simple API to do some common NLP tasks.  The
# transformers library was developed by Hugging Face : 
# https://huggingface.co/docs/transformers/index
from transformers import pipeline

# This sets up an unmasker pipeline that allows us to get BERT's Masked Language
# Model predictions for a given sentence (ie - this is where we can get it to
# play Blankety Blank with our own text).  It'll give us some predictions about
# what the missing word is, with descending levels of confidence.  We also
# specify the BERT model we want to use; here, we use the model
# 'bert-base-uncased' - this is the original BERT model.  Uncased means that it
# ignores whether letters are capital letters or lower case (so, for example,
# it would treat "Dan" and "dan" as the same word).  You can read a description
# of the model here : https://huggingface.co/bert-base-uncased
unmasker = pipeline('fill-mask', model='bert-base-uncased')

# Now let's apply the unmasker to some text.  We'll blank out one of the words
# with the token [MASK] and BERT will provide us with the five predictions for
# the missing word that it thinks most likely, in descending order of
# confidence.  Important - it's looking for punctuation too, so if you miss off
# a full stop, it might predict punctuation for your sentence (it might do that
# anyway, depending on your sentence).  Note - you can only mask out a single
# token.
unmasker("Cats are generally very [MASK].")
#unmasker("The man worked as a [MASK].")
#unmasker("The woman worked as a [MASKs].")
#unmasker("The nurse tended to the patient, and [MASK] spoke to the doctor.")

