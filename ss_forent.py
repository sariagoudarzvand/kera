from sklearn.metrics.pairwise import cosine_similarity, cosine_distances,paired_cosine_distances
import tensorflow as tf
tf.enable_eager_execution()
documents = (
"The sky is blue",
"The sun is bright",
"The sun in the sky is bright",
"We can see the shining sun, the bright sun")

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

def cosine_score(x):
	# Normalize the columns of the tensor
	# normalized_tensor = tf.math.l2_normalize(x, axis=0)
	normalized_tensor = x
	# Get the dot product between the columns
	scores = tf.cast(tf.matmul(normalized_tensor, normalized_tensor, transpose_a=True), tf.float32)
	zero_diag = tf.linalg.set_diag(tf.cast(scores, tf.float32), tf.cast(tf.zeros(tf.shape(scores)[0]), tf.float32))
	# triangular = tf.matrix_band_part(zero_diag, 0, -1)
	return zero_diag


tf_a1 = tf.Variable([    [0.0968594,   -0.0655439,    -0.6,             -0.3            ],
                         [-0.21,          -0.003356,     -0.9,            -0.08974       ],
                         [0.22,          -0.23,         - 0.103182,      -0.0000330564],
                         [-0.0609862,  0.24,           -0.614321,      -0.35         ],
                         [0.00497023,  0.25,          -0.8914037,      0.2           ],
                         [0.23,          0.457685,    0.602337,        0.11          ],
                         [0.002,       0.6,          0.826657,       0.283971        ]])
# 1.11573168e-145

def rev_entropy(x):
    def row_entropy(row):
        import tensorflow as tf
        if row is not None:
            data = tf.reshape(row, shape=[1, -1])
            num_samples = data.shape[0]
            if len(data.shape) == 1:
                num_dimensions = 1
            else:
                num_dimensions = data.shape[1]
        epsilon = tf.constant(0.000001)
        from numpy.linalg import det
        detCov = tf.linalg.det(
            tf.cast(tf.matmul(data, tf.transpose(data)), tf.float32) / tf.cast(num_samples, tf.float32))
        normalization = tf.math.pow(
            tf.cast((tf.math.multiply(2., tf.math.multiply(np.pi, tf.math.exp(1.0)))), tf.int32),
            num_dimensions)
        if detCov == 0:
            return -np.inf
        else:
            return 1 / 1 + (0.5 * tf.math.log(
                epsilon + tf.math.multiply(tf.cast(normalization, tf.float32), tf.cast(detCov, tf.float32))))

    rev = tf.map_fn(row_entropy, x, dtype=tf.float32)
    return rev
k = 2

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
# tf_a1 = np.load('testnew.npy')
# print tf_a1
ent_p = rev_entropy(tf_a1)

p = (tf_a1 + tf.abs(tf_a1)) / 2

ent_threshold = tf.reduce_mean(ent_p)
ent_mask_p = tf.greater(ent_p, ent_threshold)
print ent_mask_p
topk_bad_rows = tf.where(tf.logical_not(ent_mask_p), p, tf.zeros_like(tf_a1))
topk_bad_rows_value, ind = tf.nn.top_k(topk_bad_rows, k)
my_range = tf.expand_dims(tf.range(0, tf.shape(ind)[0]), 1)  # will be [[0], [1]]
my_range_repeated = tf.tile(my_range, [1, k ])  # will be [[0, 0], [1, 1]]
full_indices = tf.stack([my_range_repeated, ind],
                                    axis=2)  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
full_indices = tf.reshape(full_indices, [-1, 2])
P_reset = tf.sparse_to_dense(full_indices, tf.shape(tf_a1), tf.reshape(topk_bad_rows_value, [-1]), default_value=0.,
                                         validate_indices=False)
ent_bad_rows = tf.multiply(p, tf.cast(tf.reshape(tf.logical_not(ent_mask_p), shape=[-1, 1]), dtype=tf.float32))-P_reset

sum_ent_bad_rows = tf.reduce_sum(ent_bad_rows)
# print sum_ent_bad_rows
indices_ent_bad_rows = tf.where(P_reset)

# # print sum_ent_bad_rows
enegy_ent_bad_rows = tf.cast(sum_ent_bad_rows / tf.cast(tf.shape(indices_ent_bad_rows)[0], dtype=tf.float32), tf.float32)
# print enegy_ent_bad_rows
energy_matrice_ent_bad_rows = tf.scatter_nd(tf.cast(indices_ent_bad_rows, tf.int32),
                                 enegy_ent_bad_rows * tf.cast(tf.ones(shape=(tf.shape(indices_ent_bad_rows)[0])), tf.float32),
                                         shape=tf.shape(p))
# print energy_matrice_ent_bad_rows
p_final_suming_ent_bad_rows = energy_matrice_ent_bad_rows + P_reset

#for negativity part:

n = (tf_a1 - tf.abs(tf_a1))/2
print n
topk_bad_rows = tf.where(tf.logical_not(ent_mask_p), n, tf.zeros_like(tf_a1))

topk_bad_rows_value, ind = tf.nn.top_k(-topk_bad_rows, k)
my_range = tf.expand_dims(tf.range(0, tf.shape(ind)[0]), 1)  # will be [[0], [1]]
my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]
full_indices = tf.stack([my_range_repeated, ind],
                                    axis=2)  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
full_indices = tf.reshape(full_indices, [-1, 2])
P_reset = tf.sparse_to_dense(full_indices, tf.shape(tf_a1), tf.reshape(topk_bad_rows_value, [-1]), default_value=0.,
                                         validate_indices=False)
ent_bad_rows = -tf.multiply(n, tf.cast(tf.reshape(tf.logical_not(ent_mask_p), shape=[-1, 1]), dtype=tf.float32)) - P_reset
sum_ent_bad_rows = tf.reduce_sum(ent_bad_rows)
indices_ent_bad_rows = tf.where(P_reset)

# # print sum_ent_bad_rows
enegy_ent_bad_rows = tf.cast(sum_ent_bad_rows / tf.cast(tf.shape(indices_ent_bad_rows)[0], dtype=tf.float32), tf.float32)
energy_matrice_ent_bad_rows = tf.scatter_nd(tf.cast(indices_ent_bad_rows, tf.int32),
                                 enegy_ent_bad_rows * tf.cast(tf.ones(shape=(tf.shape(indices_ent_bad_rows)[0])), tf.float32),
                                         shape=tf.shape(n))
n_final_suming_ent_bad_rows = energy_matrice_ent_bad_rows + P_reset
final1 = -n_final_suming_ent_bad_rows + p_final_suming_ent_bad_rows


###########topk smallest

dim0 = p.shape[0]
a = tf.cast(tf.equal(p, 0), p.dtype)
b = tf.reshape(tf.reduce_sum(a, 1) + k, (dim0, -1))

c = tf.cast(tf.argsort(tf.argsort(p, 1), 1), p.dtype)

d = tf.logical_or(tf.less(c, b), tf.reshape(tf.logical_not(ent_mask_p), (dim0, -1)))

d = tf.multiply(tf.cast(d, p.dtype),  tf.reshape(tf.cast(ent_mask_p, p.dtype), (dim0, -1)))
final_good = tf.cast(p * tf.cast(d, p.dtype), tf.float32)

# ent_mask_p = tf.reshape(ent_mask_p, (dim0, -1))
indices = tf.where(tf.multiply(final_good, tf.cast(tf.reshape(ent_mask_p, (dim0, -1)), dtype=tf.float32)))

bad_rows = tf.cast(p * tf.cast(tf.reshape(ent_mask_p, (dim0, -1)), p.dtype), tf.float32) - final_good
sum_bad_rows = tf.reduce_sum(bad_rows)
enegy_n = tf.cast(sum_bad_rows / tf.cast(tf.shape(indices)[0], dtype=tf.float32), tf.float32)
energy_matrice_n = tf.scatter_nd(tf.cast(indices, tf.int32),
                                         enegy_n * tf.cast(tf.ones(shape=(tf.shape(indices)[0])), tf.float32),
                                         shape=tf.shape(final_good))
p_final_suming = energy_matrice_n + final_good

#########for negtive part

dim0 = n.shape[0]
n = tf.abs(n)
a = tf.cast(tf.equal(n, 0), p.dtype)
b = tf.reshape(tf.reduce_sum(a, 1) + k, (dim0, -1))
c = tf.cast(tf.argsort(tf.argsort(n, 1), 1), n.dtype)
d = tf.logical_or(tf.less(c, b), tf.reshape(tf.logical_not(ent_mask_p), (dim0, -1)))
d = tf.multiply(tf.cast(d, p.dtype),  tf.reshape(tf.cast(ent_mask_p, n.dtype), (dim0, -1)))
final_good = tf.cast(n * tf.cast(d, n.dtype), tf.float32)
# ent_mask_p = tf.reshape(ent_mask_p, (dim0, -1))
indices = tf.where(tf.multiply(final_good, tf.cast(tf.reshape(ent_mask_p, (dim0, -1)), dtype=tf.float32)))

bad_rows = tf.cast(n * tf.cast(tf.reshape(ent_mask_p, (dim0, -1)), n.dtype), tf.float32) - final_good
sum_bad_rows = tf.reduce_sum(bad_rows)
enegy_n = tf.cast(sum_bad_rows / tf.cast(tf.shape(indices)[0], dtype=tf.float32), tf.float32)
energy_matrice_n = tf.scatter_nd(tf.cast(indices, tf.int32),
                                         enegy_n * tf.cast(tf.ones(shape=(tf.shape(indices)[0])), tf.float32),
                                         shape=tf.shape(final_good))

n_final_suming = -energy_matrice_n - final_good

final2 = tf.cast(p_final_suming,tf.float32) + tf.cast(n_final_suming, tf.float32)

final = final1 + tf.cast(final2, tf.float32)






#
# d = tf.logical_or(tf.greater_equal(c, b), tf.reshape(masked_t_bad, (-1, dim0)))
# # print d
# d = tf.multiply(tf.cast(d, p.dtype),  tf.reshape(tf.cast(tf.logical_not(masked_t_bad), p.dtype), (-1, dim0)))
# # print d
# final_good = p * tf.cast(d, p.dtype)
# # print final_good
# indices_good = tf.where(tf.multiply(final_good, tf.cast(tf.logical_not(masked_t_bad), dtype=tf.float32)))
# # print indices_good
# bad_rows = tf.cast(p * tf.cast(tf.logical_not(masked_t_bad), p.dtype), tf.float32) - tf.cast(final_good, tf.float32)
# # print bad_rows
# sum_bad_rows = tf.reduce_sum(bad_rows)
# # print sum_bad_rows
# enegy_n = sum_bad_rows / tf.cast(tf.shape(indices_good)[0], dtype=tf.float32)
# # print enegy_n
# energy_matrice_n = tf.scatter_nd(tf.cast(indices_good, tf.int32),
#                                              enegy_n * tf.cast(tf.ones(shape=(tf.shape(indices_good)[0])), tf.float32),
#                                              shape=tf.shape(final_good))
#
# final_suming2 =tf.cast(energy_matrice_n, tf.float32) + tf.cast(final_good, tf.float32)
# # print final_suming2
# final = tf.cast(final_suming2 ,tf.float32)+ tf.cast(final_suming, tf.float32)
#
# # print final
# n = (tf_a1 - tf.abs(tf_a1)) / 2
# final = n + tf.cast(final, tf.float32)
# # print final