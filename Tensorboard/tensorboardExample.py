# Tensorboard Example

# command to start tensorboard ( dont froget to activarte python env )
# tensorboard --logdir="./output"

import tensorflow as tf

with tf.name_scope("Operation_A"):
    a = tf.add(1,2,name = "FirstAdd")
    a1 = tf.add(100,200,name= "a_add")
    a2 = tf.multiply(a,a1)
    
with tf.name_scope("Operation_B"):
    b = tf.add(3,4, name = "SecondAdd")
    b1 = tf.add(300,400, name= "b_add")
    b2 = tf.multiply(b,b1)

c = tf.multiply(a2,b2, name = "FinalResult")

with tf.Session() as sess:
    
    writer = tf.summary.FileWriter("./output",sess.graph)
    
    print(sess.run(c))
    
    writer.close()
    
# Histogram
    
k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# Setup a session and summary writer
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./tmp/histogram_example")

    summaries = tf.summary.merge_all()

    # Setup a loop and write the summaries to disk
    N = 400
    for step in range(N):
        
        k_val = step/float(N)
        summ = sess.run(summaries, feed_dict={k: k_val})
        writer.add_summary(summ, global_step=step)
        
    writer.close()