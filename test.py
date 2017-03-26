import tensorflow as tf

#define a variable to hold normal random values 
normal_rv = tf.Variable( tf.truncated_normal([1, 3], stddev = 0.1))
argmax_rv = tf.argmax(normal_rv, 1)

#initialize the variable
init_op = tf.initialize_all_variables()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print (sess.run(normal_rv))
    print (sess.run(argmax_rv))
