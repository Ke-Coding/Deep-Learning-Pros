import numpy as np

if __name__ == '__main__':
	ar = np.arange(1, 10).reshape(-1, 3)
	br = np.zeros((3, 3), dtype=int)
	br += 2 if True else 1

	print("ar:\n", ar)
	print("\nbr:\n", br)

	# axis: 0--x, 1--y, 2--z, etc.  "sum_y" means getting the summation of each row.
	print("\nsum ar: (axis=1, which means getting the summation of each row)\n", np.sum(ar, axis=1, keepdims=True))

	print("\nouter: (tensor-product)\n", np.outer(ar, br))	# tensor-product
	print("\ndot  : (matrix-product)\n", np.dot(ar, br))	# matrix-product
	print("\nmulti: (element-wise)\n", np.multiply(ar, br))	# element-wise

	print("\nar*br: (also element-wise)\n", ar * br)		# also element-wise
