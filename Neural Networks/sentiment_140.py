import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sys
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('ISO-8859-1')
sys.stdout = stdout

print("Finished importing")

lemmatizer = WordNetLemmatizer()
train_data = pd.read_csv('/Users/neelbakshi/Documents/Machine Learning/Datasets/Sentiment Analysis/trainingandtestdata/training.1600000.processed.noemoticon.csv')
test_data = pd.read_csv('/Users/neelbakshi/Documents/Machine Learning/Datasets/Sentiment Analysis/trainingandtestdata/testdata.manual.2009.06.14.csv')
print("Finished reading data")

print("Starting to load train_x, train_y, test_x, test_y")
train_x = train_data.iloc[:,[5]].values
train_y = train_data.iloc[:, :1].values
test_x = test_data.iloc[:,[5]].values
test_y = test_data.iloc[:, :1].values

train_x = [tweet_array[0] for tweet_array in train_x]
train_y = [polarity_array[0] for polarity_array in train_y]
test_x = [tweet_array[0] for tweet_array in test_x]
test_y = [polarity_array[0] for polarity_array in test_y]
print("Finished getting them into our own format")

try:
    with open("lexicon.pickle",'rb') as f:
        print("Found lexicon pickle")
        final_lexicon = pickle.load(f)
except:
    print("Preparing the lexicon")
    lexicon = []
    for index, line in enumerate(train_x):
        words_in_line = word_tokenize(line)
        for word in words_in_line:
            lexicon.append(lemmatizer.lemmatize(word.lower()))
        print("Line finished: {0}").format(index+1)

    print("Doing word count")
    word_count = Counter(lexicon)

    final_lexicon = []
    for word in word_count:
        if 10000 > word_count[word] > 50:
            final_lexicon.append(word)
    print("Finished preparing the lexicon")
    with open('lexicon.pickle','wb') as f:
            pickle.dump(final_lexicon,f)

def convert_test_labels(train_y):
    labels_y = []
    for y in train_y:
        if y == 0:
            labels_y.append([0,1])
        else:
            labels_y.append([1,0])
    return labels_y

print("Formatting our test and train labels")
train_y = convert_test_labels(train_y)
test_y = convert_test_labels(test_y)

print("Final prepocessing step begins")
train_set = [(x,y) for x,y in zip(train_x, train_y)]
test_set = [(x,y) for x,y in zip(test_x, test_y)]

def convert_to_feature_set(tweet, lexicon, index_of_tweet):
    #print("Processing tweet: {0}").format(tweet)
    print("Processing tweet no: {0}").format(index_of_tweet)
    feature = np.zeros(len(lexicon))
    words_in_tweet = word_tokenize(tweet)
    for word in words_in_tweet:
        try:
            index_of_word_in_lexicon = lexicon.index(word)
            feature[index_of_word_in_lexicon] += 1
        except:
            pass
    return feature


class Network:

    def __init__(self, structure, input_placeholder, output_placeholder):
        self.structure = structure
        self.input_placeholder = input_placeholder
        self.output_placeholder = output_placeholder
        layers = []
        for current_nodes, previous_nodes in zip(self.structure[1:], self.structure[:-1]):
            weights = tf.Variable(tf.random_normal([previous_nodes, current_nodes]))
            biases = tf.Variable(tf.random_normal([current_nodes]))
            layers.append({'weights': weights, 'biases': biases})

        layer_output = self.input_placeholder
        for layer in layers[:-1]:
            layer_output = tf.add(tf.matmul(layer_output, layer['weights']), layer['biases'])
            layer_output = tf.nn.relu(layer_output)
        self.final_output = tf.add(tf.matmul(layer_output, layers[-1]['weights']), layers[-1]['biases'])
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.output_placeholder, logits=self.final_output))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        print("Finished network initialization")

    def fit(self, train_data, epochs, batch_size, test_data):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("Training begins...")
            for epoch in xrange(epochs):
                epoch_loss = 0
                for batch_no in range(int(len(train_data)/batch_size)):
                    print("Starting batch {0}").format(batch_no)
                    epoch_data = train_data[(batch_no*batch_size):((batch_no + 1)*batch_size)]
                    print("Size of batch: {0}").format(len(epoch_data))
                    epoch_x = [convert_to_feature_set(x[0], final_lexicon, index) for index, x in enumerate(epoch_data)]
                    epoch_y = [y[1] for y in epoch_data]
                    c, _ = sess.run([self.cost, self.optimizer], feed_dict={self.input_placeholder:epoch_x, self.output_placeholder:epoch_y})
                    epoch_loss += c
                print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(self.final_output, 1), tf.argmax(self.output_placeholder, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print("Feeding test data in the network")
            input_test = [convert_to_feature_set(x[0], final_lexicon, index) for index, x in enumerate(test_data)]
            output_test = [y[1] for y in test_data]
            print('Accuracy:',accuracy.eval({self.input_placeholder:input_test, self.output_placeholder:output_test}))


print("Creating network")
network = Network([len(final_lexicon), 500, 300, 200, 2], tf.placeholder('float'), tf.placeholder('float'))
print("Network training and testing begins")
network.fit(train_set, 10, 500, test_set)
