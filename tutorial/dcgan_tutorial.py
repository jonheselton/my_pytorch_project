# DCGAN
# Generative Adversarial Network - A framework for training a DLM that captures
#   the distribution of the training data inorder to generate new data using
#   that distribution It uses a pair of models, a Generator and a Discriminator.
#   The generator creates an image trying to replicate the training data asd
#   the discriminator tries to identify if the image is part of the training
#   data, or if it is generated by the generator.Equilibrum is reached when the
#   generator produces images consistently that the discriminator is "guessing"
#   at a 50$  success/failure rate.
# Distribution -
# Discriminator - Takes input data and outputs the probability, as a scalar,
#   that the data is generated by the Generator or is from the training data.
#   It should return a high output if the data is from the training set, or low
#   if it is from the generator.  It can be thought of as a Binary Classifier.
