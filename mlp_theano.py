import theano
import theano.tensor as T 
import numpy as np 

class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None,
    		 activation=T.tanh):

    	self.input = input
    	if W is None:
    	    W_values = np.asarray(
    	        np.random.uniform(
    	            low=-np.sqrt(6. / (n_in + n_out)),
    	            high=np.sqrt(6. / (n_in + n_out)),
    	            size=(n_in, n_out)
    	        ),
    	        dtype=theano.config.floatX
    	    )
    	    if activation == theano.tensor.nnet.sigmoid:
    	        W_values *= 4

    	    W = theano.shared(value=W_values, name='W', borrow=True)

    	if b is None:
    	    b_values = np.zeros((n_out,), dtype=theano.config.floatX)
    	    b = theano.shared(value=b_values, name='b', borrow=True)

    	self.W = W
    	self.b = b

    	lin_output = T.dot(input, self.W) + self.b
    	self.output = (
    	    lin_output if activation is None
    	    else activation(lin_output)
    	)
    	# parameters of the model
    	self.params = {'W':self.W, 'b':self.b}

class MLP(object):
	def __init__(self, sizes, input_dim, output_dim):
		self.layers = len(sizes) + 1
				
		in_dim = [input_dim] + sizes
		out_dim = sizes + [output_dim]
		x = T.dvector('x')
		y = T.dvector('y')
		self.hyp_params = []
		for i, (r,c) in enumerate(zip(in_dim,out_dim)):
			if i == 0:
				obj = HiddenLayer(x, r, c)
			else:
				obj = HiddenLayer(obj.output,r,c)
			self.hyp_params.append(obj.params)

		

		yhat = obj.output

		prediction = T.argmax(yhat)
		self.predict = theano.function([x],[yhat])
		o_error = T.sum(T.sqr(yhat - y))
		# o_error = T.sum(T.nnet.categorical_crossentropy(yhat, y))
		updates = []
		learning_rate = T.scalar('learning_rate')
		for param in self.hyp_params:
			updates.append((param['W'], param['W'] - learning_rate * T.grad(o_error,param['W'])))
			updates.append((param['b'], param['b'] - learning_rate * T.grad(o_error,param['b'])))

		self.train_step = theano.function([x,y,learning_rate],[o_error],
						updates = updates)



obj = MLP([2,2],2,1)
theano.printing.pydotprint(obj.predict, outfile="logreg_pydotprint_prediction_mlp_new.png", var_with_name_simple=True)
exit()
x = [[0.0 ,0.0], [0.0, 1.0], [1.0 ,0.0],[1.0 ,1.0]]
y = [[0.0], [1.0], [1.], [0.0]]
# x = np.array([0.0, 1.0])
# print x.shape
# exit()
# y = np.array([.99])
# k = obj.train_step(x,y,0.1)
for i in range(10000):
	# print i
	for i in range(len(x)):
		ix = np.array(x[i])
		# print ix.shape
		# exit()
		iy = np.array(y[i])
		loss = obj.train_step(ix,iy,0.1)[0]
		print loss
		


# print obj.predict(np.array([0.0, 0.0]))[0]
# print obj.predict(np.array([0.0, 1.0]))[0]
# print obj.predict(np.array([1.0, 0.0]))[0]
# print obj.predict(np.array([1.0, 1.0]))[0]






