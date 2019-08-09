import tempfile
import tensorflow as tf

with open("text1.txt", "w") as file:
    file.write("File1, line1.\n")
    file.write("File2, line2.\n")

with open("text2.txt", "w") as file:
    file.write("File2, line1.\n")
    file.write("File2, line2.\n")

# generate dataset from text files
input_files = ["text1.txt", "text2.txt"]
dataset = tf.data.TextLineDataset(input_files)

# define iterator
iterator = dataset.make_one_shot_iterator()

# get_next() returns one line in text file
x = iterator.get_next()
with tf.Session() as sess:
    for i in range(4):
        print(sess.run(x))
