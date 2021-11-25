'''
    Softmax回归
'''
# coding:utf-8

import tensorflow as tf  # 导入tensorflow
from tensorflow.examples.tutorials.mnist import input_data  # 导入MNIST教学的模
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 读入MNIST数据
'''
    1. 语法：tf.placeholder( dtype, shape=None, name ) dtype是数据类型;shape是数据形状，默认一维，也可二维;name可有可无
    2. palceholder只暂时存储变量，传值过程在sess.run()中进行
'''

# 创建x，x是一个占位符（placeholder），用于得到传递进来的待识别的训练图片
x = tf.placeholder(tf.float32, [None, 784])

# W是Softmax模型的参数，将一个784维的输入转换为一个10维的输出
# 在TensorFlow中，变量的参数用tf.Variable表示
W = tf.Variable(tf.zeros([784, 10]))  # 该操作返回一个带有形状shape的类型为dtype张量,并且所有元素都初始化为0
# b是又一个Softmax模型的参数，我们一般叫做“偏置项”（bias）
b = tf.Variable(tf.zeros([10]))

# y=softmax(Wx + b)，y表示模型的输出
y = tf.nn.softmax(tf.matmul(x, W) + b)  # tf.matmual()表示两个矩阵相乘

# y_是实际的图像标签，同样以占位符表示
y_ = tf.placeholder(tf.float32, [None, 10])

# 至此，我们得到了两个重要的Tensor：y和y_。
# y是模型的输出，y_是实际的图像标签，不要忘了y_是独热表示的
# 下面我们就会根据y和y_构造损失

# 根据y, y_构造交叉熵损失
'''
    reduce_sum():计算一个张量的各个维度上元素的总和
    reduce_mean():计算张量的各个维度上的元素的平均值
'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))  # y * e^y

# 有了损失，我们就可以用随机梯度下降针对模型的参数（W和b）进行优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
'''
    这一行代码实际上是用来往计算图上添加一个新操作,其中包括计算梯度,计算每个参数的步长变化,并且计算出新的参数值.
    返回的train_step操作对象,在运行时会使用梯度下降来更新参数.因此,整个模型的训练可以通过反复地运行train_step来完成.
'''
# 创建一个Session，只有在Session中才能运行优化步骤train_step
sess = tf.InteractiveSession()
# 运行之前必须要初始化所有变量，分配内存
tf.global_variables_initializer().run()
print('start training...')

# 进行1000步梯度下降
for _ in range(1000):
    # 在mnist.train中取100个训练数据
    # batch_xs是形状为(100, 784)的图像数据，batch_ys是形如(100, 10)的实际标签
    # batch_xs, batch_ys对应着两个占位符x和y_
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 在Session中运行train_step，运行时要传入占位符的值
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # feed_dict给使用placeholder创建出来的tensor赋值
'''
    eg:[True,False,True,True]可以用[1,0,1,1]表示，精度为0.75
       0.75 = 3/4
'''

# 正确的预测结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # equal()逐元素比较是否相等,返回结一个比较结果（Ture/False）矩阵
# 计算预测准确率，它们都是Tensor
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 张量数据类型转换，将布尔型转换为float32
# 在Session中运行Tensor可以得到Tensor的值
# 这里是获取最终模型的正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # 0.9185