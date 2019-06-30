from typing import List

import numpy as np
import lr_utils


class LR_NN:
	"""
	m_train(209), m_test(50), m_pix(64), m_flat(3*64*64)

	train_set_x_orig:	(m_train, m_pix, m_pix, 3)
	train_set_x:		(m_flat, m_train)
	train_set_y:		(1, m_train)
	test_set_x_orig:	(m_test, m_pix, m_pix, 3)
	test_set_x:		(m_flat, m_test)
	test_set_y:		(1, m_test)
	
	w, dw:			(m_flat, 1)
	b, db:			scalar
	"""
	
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
	m_train = train_set_x_orig.shape[0]
	m_test = test_set_x_orig.shape[0]
	m_flat = 3 * test_set_x_orig.shape[1]**2
	
	train_set_x = train_set_x_orig.reshape(m_train, m_flat).T / 255	# .reshape(-1, m_train) != .reshape(m_train, -1).T  !!!!!(write on a paper then see how they differ)
	test_set_x = test_set_x_orig.reshape(m_test, m_flat).T / 255
	
	w, b, dw, db = [], [], [], []

	def __init__(self):
		pass
	
	def propagate(self) -> float: 				# return cost -- a scalar, negative log-likelihood cost for logistic regression
		m_data = self.train_set_x.shape[1]		# number of examples
		
		# forward propagation
		A = lr_utils.sigmoid(np.dot(self.w.T, self.train_set_x) + self.b)  # activation: (1, m_train), the same as self.train_set_y's
		cost: float = np.sum(self.train_set_y * np.log(A) + (1 - self.train_set_y) * np.log(1 - A)) / -m_data  # cost: (1, 1)

		# backward propagation
		self.dw = np.dot(self.train_set_x, (A - self.train_set_y).T) / m_data	# dw: (m_flat, 1), the same as w's
		self.db = np.sum(A - self.train_set_y) / m_data				# db: scalar
		
		return cost
	
	def gradient_decent(self, n_iter, n_check, l_rate, printable=False) -> List[float]:
		self.w, self.b = np.zeros((self.m_flat, 1), dtype=float), float(0.)	# initialization
		costs: List[float] = []

		for i in range(n_iter):
			
			# propagation
			cost = self.propagate()
			
			# gradient decent
			self.w -= l_rate * self.dw
			self.b -= l_rate * self.db
			
			# record
			if i % n_check == 0:
				costs.append(cost)
				if printable:
					print("Cost after iteration %i: %f" % (i, cost))

		return costs
	
	def predict(self, X):	# X: (m_flat, m_cur), w: (m_flat, 1)
		A = lr_utils.sigmoid(np.dot(self.w.T, X) + self.b)  # activation: (1, m_cur)
		return np.rint(A)
	
	def train(self, n_iter=4000, n_check=100, l_rate=0.005, printable=False):
		
		costs = self.gradient_decent(n_iter, n_check, l_rate, printable)
		
		train_predict = self.predict(self.train_set_x)	# train_predict: (1, m_train), the same as self.train_set_y's
		test_predict = self.predict(self.test_set_x)	# test_predict: (1, m_test), the same as self.test_set_y's
		
		# print("ans: y = " + str(self.train_set_y[:, show_index][0]) + ", it's a '" + self.classes[np.squeeze(self.train_set_y[:, show_index])].decode("utf-8") + "' picture.")
		# for i in range(self.m_test):
		# 	if abs(test_predict[0][i]-self.test_set_y[0][i]) > 1e-4:
		# 		plt.imshow(self.test_set_x_orig[i])
		
		print("train accuracy: {} %".format(100 - np.mean(np.abs(train_predict - self.train_set_y)) * 100))
		print("test accuracy: {} %".format(100 - np.mean(np.abs(test_predict - self.test_set_y)) * 100))
		
		# plt.show()
		
		d = {"costs": np.array(costs).reshape(4, -1),
			 "test_predict": test_predict,
			 "train_predict": train_predict,
			 "weight": self.w,
			 "bias": self.b,
			 "learning_rate": l_rate,
			 "num_of_iterations": n_iter}

		return d
