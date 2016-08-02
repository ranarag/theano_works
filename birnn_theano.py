import numpy as np
import theano
import theano.tensor as T
import theano.typed_list
import operator


class BIRNN(object):

    def __init__(self, word_dim, hidden_dim=10, sequence_length=3):
    # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.seq_length = sequence_length

        # Randomly initialize the network parameters
        Wxh_f = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        Why_f = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        Whh_f = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        # Theano: Created shared variables
        self.Wxh_f = theano.shared(name='Wxh_f', value=Wxh_f.astype(theano.config.floatX))
        self.Why_f = theano.shared(name='Why_f', value=Why_f.astype(theano.config.floatX))
        self.Whh_f = theano.shared(name='Whh_f', value=Whh_f.astype(theano.config.floatX))

        Wxh_b = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        # Whh_fb = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
        Why_b = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        Whh_b = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        # Theano: Created shared variables
        self.Wxh_b = theano.shared(name='Wxh_b', value=Wxh_b.astype(theano.config.floatX))
        self.Why_b = theano.shared(name='Why_b', value=Why_b.astype(theano.config.floatX))
        self.Whh_b = theano.shared(name='Whh_b', value=Whh_b.astype(theano.config.floatX))
        # self.Whh_fb = theano.shared(name='Whh_fb', value=Whh_fb.astype(theano.config.floatX))

        bhh_f = np.zeros((hidden_dim,), dtype=theano.config.floatX)
        bhh_b = np.zeros((hidden_dim,), dtype=theano.config.floatX)
        bhy = np.zeros((word_dim,), dtype=theano.config.floatX)

        self.bhh_f = theano.shared(name='bhh_f', value=bhh_f)
        self.bhh_b = theano.shared(name='bhh_b', value=bhh_b)
        self.bhy = theano.shared(name='bhy', value=bhy)

        self.theano_build()

    def theano_init(self):
        x = T.ivector('x')
        y = T.ivector('y')

        def forward_pass_f(x_t, st_prev, Ul, Wl, bhl):
            st = T.tanh(Ul[:,x_t] + Wl.dot(st_prev) + bhl)
            return st

        Ul, Wl, bhl = self.Wxh_f, self.Whh_f, self.bhh_f
        hf_states, updates = theano.scan(
            forward_pass_f,
            sequences=x,
            outputs_info=[dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[Ul, Wl, bhl])

        # for second layer
        def forward_passb(a_x, a_s_t, U, W, bh):
            # print T.shape(x_t)
            # print T.shape(U)
            # print st_prev.ndim
            # exit()
            s_t = T.tanh(U[:,a_x] + W.dot(a_s_t) + bh)
            # o_t = T.nnet.softmax(V.dot(s_t) + by)
            return s_t



        U, W, bh = self.Wxh_b, self.Whh_b, self.bhh_b
        hb_states, updates = theano.scan(
            forward_passb,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, W, bh]
        )
        hb_states = hb_states[::-1]
        o = T.nnet.softmax(self.Why_f.dot(hf_states)+
                           self.Why_b.dot(hb_states)+
                           self.by)
        prediction = T.argmax(o, axis=1)

        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        params = [self.Wxh_f,self.Whh_f,self.Wxh_b,self.Why_b,self.Whh_b, self.bhh_f,self.bhh_b,self.bhy]
        gparams = [T.grad(o_error, param) for param in params]
        learning_rate = T.scalar('learning_rate')
        clipped_gparams = [T.clip(gparam, T.ones_like(gparam)*-5.0, T.ones_like(gparam)*5.0) for gparam in gparams]

        updates = [(param, param - learning_rate*gparam) for param,gparam in zip(params,clipped_gparams)]

        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.calc_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], clipped_gparams)


        self.train_step = theano.function([x,y,learning_rate], [o_error],
                                          updates=updates)


