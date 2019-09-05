'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
import os
import numpy as np
from keras.layers import Dense
from keras.callbacks import Callback
import keras.backend as K
from keras.engine import Layer
import tensorflow as tf
from keras import initializers
import warnings
# import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

from ..testing.visualize import heatmap
from .op_utils import unitmatrix

def contractive_loss(model, lam=1e-4):
    def loss(y_true, y_pred):
        ent_loss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

        W = K.variable(value=model.encoder.get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = model.encoder.output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        return ent_loss + contractive
    return loss


def weighted_binary_crossentropy(feature_weights):
    def loss(y_true, y_pred):
        # try:
        #     x = K.binary_crossentropy(y_pred, y_true)
        #     # y = tf.Variable(feature_weights.astype('float32'))
        #     # z = K.dot(x, y)
        #     y_true = tf.pow(y_true + 1e-5, .75)
        #     y2 = tf.div(y_true, tf.reshape(K.sum(y_true, 1), [-1, 1]))
        #     z = K.sum(tf.mul(x, y2), 1)
        # except Exception as e:
        #     print e
        #     import pdb;pdb.set_trace()
        # return z
        return K.dot(K.binary_crossentropy(y_pred, y_true), K.variable(feature_weights.astype('float32')))
    return loss

class KCompetitive(Layer):
    '''Applies K-Competitive layer.

    # Arguments
    '''

    def __init__(self, topk, ctype, **kwargs):
        self.topk = topk
        self.ctype = ctype
        # self.cmodel = cmodel
        self.uses_learning_phase = True
        self.supports_masking = True
        super(KCompetitive, self).__init__(**kwargs)

    def call(self, x):
        if self.ctype == 'ksparse':
            return K.in_train_phase(self.kSparse(x, self.topk), x)
        elif self.ctype == 'kcomp':
            return K.in_train_phase(self.k_comp_tanh(x, self.topk), x)
        else:
            warnings.warn("Unknown ctype, using no competition.")
            return x

    def get_config(self):
        config = {'topk': self.topk, 'ctype': self.ctype}
        base_config = super(KCompetitive, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def k_comp_tanh(self, tf_a1, topk, factor=6.26):
        # import tensorflow as tf
        #
        # x = tf_a1
        # print 'run k_comp_tanh'
        # dim = int(x.get_shape()[1])
        # # batch_size = tf.to_float(tf.shape(x)[0])
        # if topk > dim:
        #     warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
        #     topk = dim
        #
        # P = (x + tf.abs(x)) / 2
        # N = (x - tf.abs(x)) / 2
        #
        # values, indices = tf.nn.top_k(P,
        #                                   topk / 2)  # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]
        # # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        # my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        # my_range_repeated = tf.tile(my_range, [1, topk / 2])  # will be [[0, 0], [1, 1]]
        # full_indices = tf.stack([my_range_repeated, indices],
        #                             axis=2)  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        # full_indices = tf.reshape(full_indices, [-1, 2])
        # P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0.,
        #                                  validate_indices=False)
        #
        # values2, indices2 = tf.nn.top_k(-N, topk - topk / 2)
        # my_range = tf.expand_dims(tf.range(0, tf.shape(indices2)[0]), 1)
        # my_range_repeated = tf.tile(my_range, [1, topk - topk / 2])
        # full_indices2 = tf.stack([my_range_repeated, indices2], axis=2)
        # full_indices2 = tf.reshape(full_indices2, [-1, 2])
        # N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(values2, [-1]), default_value=0.,
        #                                  validate_indices=False)
        #
        # # 1)
        # # res = P_reset - N_reset
        # # tmp = 1 * batch_size * tf.reduce_sum(x - res, 1, keep_dims=True) / topk
        #
        # # P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)
        # # N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)
        #
        # # 2)
        # # factor = 0.
        # # factor = 2. / topk
        # P_tmp = factor * tf.reduce_sum(P - P_reset, 1, keep_dims=True)  # 6.26
        # N_tmp = factor * tf.reduce_sum(-N - N_reset, 1, keep_dims=True)
        # P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, P_tmp), [-1]),
        #                                  default_value=0., validate_indices=False)
        # N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, N_tmp), [-1]),
        #                                  default_value=0., validate_indices=False)
        #
        # res = P_reset - N_reset
        #
        # return res
        # else:
###2
            def cosine_score(x):
                # Normalize the columns of the tensor
                # normalized_tensor = tf.math.l2_normalize(x, axis=0)
                # Get the dot product between the columns
                scores = tf.cast(tf.matmul(x, x, transpose_a=True), tf.float64)
                zero_diag = tf.linalg.set_diag(tf.cast(scores, tf.float64), tf.cast(tf.zeros(tf.shape(scores)[0]), tf.float64))
                # triangular = tf.matrix_band_part(zero_diag, 0, -1)
                return zero_diag

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
            import tensorflow as tf


            # k = topk
            # sim_topics = cosine_score(tf_a1)
            # threshold = tf.reduce_mean(sim_topics)
            # masked_bad_col = tf.greater(tf.count_nonzero(tf.greater(sim_topics, threshold), axis=1), topk)
            #
            # p = (tf_a1 + tf.abs(tf_a1)) / 2
            # dim0 = p.shape[1]
            # a = tf.cast(tf.equal(p, 0), p.dtype)
            # b = tf.reshape(tf.reduce_sum(a, 0) + topk, (-1, dim0))
            # c = tf.cast(tf.argsort(tf.argsort(p, 0), 0), p.dtype)
            # d = tf.logical_or(tf.less_equal(c, b), tf.reshape(tf.logical_not(masked_bad_col), (-1, dim0)))
            # d = tf.multiply(tf.cast(d, p.dtype), tf.reshape(tf.cast(masked_bad_col, p.dtype), (-1, dim0)))
            # final_good = p * tf.cast(d, p.dtype)
            #
            # masked_bad_col = tf.reshape(masked_bad_col, (-1, dim0))
            # indices_good = tf.where(tf.multiply(final_good, tf.cast(masked_bad_col, dtype=tf.float32)))
            #
            # bad_rows = tf.cast(p * tf.cast(masked_bad_col, p.dtype), tf.float32) - final_good
            # sum_bad_rows = tf.reduce_sum(bad_rows)
            #
            # enegy_n = 6 * (sum_bad_rows+1 / tf.cast(tf.shape(indices_good)[0], dtype=tf.float32) + 1)
            #
            # energy_matrice_n = tf.scatter_nd(tf.cast(indices_good, tf.int32),
            #                                  enegy_n * tf.cast(tf.ones(shape=(tf.shape(indices_good)[0])), tf.float32),
            #                                  shape=tf.shape(final_good))
            # final_suming = energy_matrice_n + final_good
            #
            # ###for good columns make stronger the stronger ones
            #
            # d = tf.logical_or(tf.greater_equal(c, b), tf.reshape(masked_bad_col, (-1, dim0)))
            #
            # d = tf.multiply(tf.cast(d, p.dtype), tf.reshape(tf.cast(tf.logical_not(masked_bad_col), p.dtype), (-1, dim0)))
            #
            # final_good = p * tf.cast(d, p.dtype)
            #
            # indices_good = tf.where(tf.multiply(final_good, tf.cast(tf.logical_not(masked_bad_col), dtype=tf.float32)))
            #
            # bad_rows = tf.cast(p * tf.cast(tf.logical_not(masked_bad_col), p.dtype), tf.float32) - final_good
            #
            # sum_bad_rows = tf.reduce_sum(bad_rows)
            # enegy_n = 6 * (sum_bad_rows +1 / tf.cast(tf.shape(indices_good)[0], dtype=tf.float32) + 1)
            # energy_matrice_n = tf.scatter_nd(tf.cast(indices_good, tf.int32),
            #                                  enegy_n * tf.cast(tf.ones(shape=(tf.shape(indices_good)[0])), tf.float32),
            #                                  shape=tf.shape(final_good))
            #
            # final_suming2 = energy_matrice_n + final_good
            #
            # final = final_suming2 + final_suming
            #
            # n = (tf_a1 - tf.abs(tf_a1)) / 2
            # final = n + final
            #
            # final = tf.where(tf.is_nan(final), tf.ones_like(final) * 0.000001,
            #                  final)
            # return final

###################with ent the same idea

            k = topk/2

            ent_p = rev_entropy(tf_a1)

            p = (tf_a1 + tf.abs(tf_a1)) / 2

            ent_threshold = tf.reduce_mean(ent_p)
            ent_mask_p = tf.greater(ent_p, ent_threshold)
            # topk_bad_rows = tf.where(ent_mask_p, p, tf.zeros_like(tf_a1))

            topk_bad_rows = tf.where(tf.logical_not(ent_mask_p), p, tf.zeros_like(tf_a1))
            topk_bad_rows_value, ind = tf.nn.top_k(topk_bad_rows, k)
            my_range = tf.expand_dims(tf.range(0, tf.shape(ind)[0]), 1)  # will be [[0], [1]]
            my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]
            full_indices = tf.stack([my_range_repeated, ind],
                                    axis=2)  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
            full_indices = tf.reshape(full_indices, [-1, 2])
            P_reset = tf.sparse_to_dense(full_indices, tf.shape(tf_a1), tf.reshape(topk_bad_rows_value, [-1]), default_value=0.,
                                         validate_indices=False)
            #cchange here as well
            ent_bad_rows = tf.multiply(p, tf.cast(tf.reshape(tf.logical_not(ent_mask_p), shape=[-1, 1]), dtype=tf.float32)) - P_reset

            sum_ent_bad_rows = tf.reduce_sum(ent_bad_rows)
            # print sum_ent_bad_rows
            indices_ent_bad_rows = tf.where(P_reset)

            # # print sum_ent_bad_rows
            enegy_ent_bad_rows = 256 * tf.cast((1 + sum_ent_bad_rows )/ tf.cast((1 + tf.shape(indices_ent_bad_rows)[0]), dtype=tf.float32),
                                         tf.float32)
            # print enegy_ent_bad_rows
            energy_matrice_ent_bad_rows = tf.scatter_nd(tf.cast(indices_ent_bad_rows, tf.int32),
                                                        enegy_ent_bad_rows * tf.cast(
                                                            tf.ones(shape=(tf.shape(indices_ent_bad_rows)[0])), tf.float32),
                                                        shape=tf.shape(p))

            p_final_suming_ent_bad_rows = energy_matrice_ent_bad_rows + P_reset

            # for negativity part:

            n = (tf_a1 - tf.abs(tf_a1)) / 2
            # topk_bad_rows = tf.where(ent_mask_p, n, tf.zeros_like(tf_a1))
            #
            # topk_bad_rows_value, ind = tf.nn.top_k(-topk_bad_rows, k)
            # my_range = tf.expand_dims(tf.range(0, tf.shape(ind)[0]), 1)  # will be [[0], [1]]
            # my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]
            # full_indices = tf.stack([my_range_repeated, ind],
            #                         axis=2)  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
            # full_indices = tf.reshape(full_indices, [-1, 2])
            # P_reset = tf.sparse_to_dense(full_indices, tf.shape(tf_a1), tf.reshape(topk_bad_rows_value, [-1]), default_value=0.,
            #                              validate_indices=False)
            # ent_bad_rows = -tf.multiply(n, tf.cast(tf.reshape(ent_mask_p, shape=[-1, 1]),
            #                                        dtype=tf.float32)) - P_reset
            # sum_ent_bad_rows = tf.reduce_sum(ent_bad_rows)
            # indices_ent_bad_rows = tf.where(P_reset)
            #
            # # # print sum_ent_bad_rows
            # enegy_ent_bad_rows = tf.cast((sum_ent_bad_rows + 1) / tf.cast((tf.shape(indices_ent_bad_rows)[0]+1), dtype=tf.float32),
            #                              tf.float32)
            # energy_matrice_ent_bad_rows = tf.scatter_nd(tf.cast(indices_ent_bad_rows, tf.int32),
            #                                             enegy_ent_bad_rows * tf.cast(
            #                                                 tf.ones(shape=(tf.shape(indices_ent_bad_rows)[0])), tf.float32),
            #                                             shape=tf.shape(n))
            # n_final_suming_ent_bad_rows = energy_matrice_ent_bad_rows + P_reset
            # final1 = -n_final_suming_ent_bad_rows + p_final_suming_ent_bad_rows

            ###########topk smallest

            dim0 = tf.shape(p)[0]
            a = tf.cast(tf.equal(p, 0), p.dtype)

            b = tf.reshape(tf.reduce_sum(a, 1) + k, (dim0, -1))

            c = tf.cast(tf.argsort(tf.argsort(p, 1), 1), p.dtype)
            #it has to be ent_mask only
            d = tf.logical_or(tf.less(c, b), tf.reshape(tf.logical_not(ent_mask_p), (dim0, -1)))
            #chage it here
            d = tf.multiply(tf.cast(d, p.dtype), tf.reshape(tf.cast(ent_mask_p, p.dtype), (dim0, -1)))
            final_good = tf.cast(p * tf.cast(d, p.dtype), tf.float32)

            indices = tf.where(tf.multiply(final_good, tf.cast(tf.reshape(ent_mask_p, (dim0, -1)), dtype=tf.float32)))

            bad_rows = tf.cast(p * tf.cast(tf.reshape(ent_mask_p, (dim0, -1)), p.dtype), tf.float32) - final_good

            sum_bad_rows = tf.reduce_sum(bad_rows)

            enegy_n = 256 * tf.cast((1+sum_bad_rows )/ tf.cast((1 + tf.shape(indices)[0]), dtype=tf.float32), tf.float32)

            energy_matrice_n = tf.scatter_nd(tf.cast(indices, tf.int32),
                                             enegy_n * tf.cast(tf.ones(shape=(tf.shape(indices)[0])), tf.float32),
                                             shape=tf.shape(final_good))

            p_final_suming = energy_matrice_n + final_good

    #########for negtive part

            # dim0 = tf.shape(n)[0]
            # n = tf.abs(n)
            # a = tf.cast(tf.equal(n, 0), p.dtype)
            # b = tf.reshape(tf.reduce_sum(a, 1) + k, (dim0, -1))
            # c = tf.cast(tf.argsort(tf.argsort(n, 1), 1), n.dtype)
            # d = tf.logical_or(tf.less(c, b), tf.reshape(ent_mask_p, (dim0, -1)))
            # d = tf.multiply(tf.cast(d, p.dtype), tf.reshape(tf.cast(tf.logical_not(ent_mask_p), n.dtype), (dim0, -1)))
            # final_good = tf.cast(n * tf.cast(d, n.dtype), tf.float32)
            # # ent_mask_p = tf.reshape(ent_mask_p, (dim0, -1))
            # indices = tf.where(tf.multiply(final_good, tf.cast(tf.reshape(tf.logical_not(ent_mask_p), (dim0, -1)), dtype=tf.float32)))
            #
            # bad_rows = tf.cast(n * tf.cast(tf.reshape(tf.logical_not(ent_mask_p), (dim0, -1)), n.dtype), tf.float32) - final_good
            # sum_bad_rows = tf.reduce_sum(bad_rows)
            # enegy_n = tf.cast((sum_bad_rows +1)/ tf.cast((tf.shape(indices)[0] +1), dtype=tf.float32), tf.float32)
            # energy_matrice_n = tf.scatter_nd(tf.cast(indices, tf.int32),
            #                                  enegy_n * tf.cast(tf.ones(shape=(tf.shape(indices)[0])), tf.float32),
            #                                  shape=tf.shape(final_good))
            #
            # n_final_suming = - energy_matrice_n - final_good
            # final2 = tf.cast(p_final_suming, tf.float32) + tf.cast(n_final_suming, tf.float32)
            #
            # final = final1 + tf.cast(final2, tf.float32)
            final = p_final_suming_ent_bad_rows + p_final_suming + n
            return final


####3
        # threshold = 4
        # sim_topics = cosine_score(tf_a1)
        # mask_sim_topic = tf.greater(sim_topics, threshold)
        #
        # ent_p = rev_entropy(tf_a1)
        # ent_mask_p = tf.greater(ent_p, threshold)
        # p_good_rows = tf.where(ent_mask_p, tf_a1, tf.zeros_like(tf_a1))
        #
        # bad_rows = tf.where(tf.less(ent_p, threshold), tf_a1, tf.zeros_like(tf_a1))
        #
        # p_bad_rows = (bad_rows + tf.abs(bad_rows)) / 2
        # p_sum_bad_rows = tf.reduce_sum(p_bad_rows)
        # p_very_good_rows = (p_good_rows + tf.abs(p_good_rows)) / 2
        # number_of_good_items = tf.where(p_very_good_rows)
        # enegy = 6.3 * p_sum_bad_rows / tf.cast(tf.shape(number_of_good_items)[0] + 1, dtype=tf.float32)
        # energy_matrice = tf.scatter_nd(tf.cast(number_of_good_items, tf.int32),
        #                                enegy * tf.ones(shape=(tf.shape(number_of_good_items)[0])),
        #                                shape=tf.shape(tf_a1))
        #
        # result_p = p_very_good_rows + energy_matrice
        #
        # ####story for the negative weights
        # ent_n = rev_entropy(tf_a1)
        # ent_mask_n = tf.greater(ent_n, threshold)
        # n_good_rows = tf.where(ent_mask_n, tf_a1, tf.zeros_like(tf_a1))
        # n_bad_rows = tf.where(tf.less(ent_n, threshold), tf_a1, tf.zeros_like(tf_a1))
        #
        # n_bad_rows = (n_bad_rows - tf.abs(n_bad_rows)) / 2
        #
        # n_sum_bad_rows = tf.reduce_sum(n_bad_rows)
        #
        # n_very_good_rows = (n_good_rows - tf.abs(n_good_rows)) / 2
        #
        # n_number_of_good_items = tf.where(n_very_good_rows)
        # enegy_n = 6.3 * n_sum_bad_rows / tf.cast(tf.shape(n_number_of_good_items)[0] + 1, dtype=tf.float32)
        # energy_matrice_n = tf.scatter_nd(tf.cast(n_number_of_good_items, tf.int32),
        #                                  enegy_n * tf.ones(shape=(tf.shape(n_number_of_good_items)[0])),
        #                                  shape=tf.shape(tf_a1))
        # result_n = n_very_good_rows + energy_matrice_n
        # res = result_p - result_n
        # return res

        # def rev_entropy(x):
        #     def log10(x):
        #         import tensorflow as tf
        #         numerator = tf.log(x)
        #         denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        #         return numerator / denominator
        #     def row_entropy(row):
        #         import tensorflow as tf
        #         if row is not None:
        #             data = tf.reshape(row, shape=[1, -1])
        #             num_samples = data.shape[0]
        #             if len(data.shape) == 1:
        #                 num_dimensions = 1
        #             else:
        #                 num_dimensions = data.shape[1]
        #         epsilon = tf.constant(0.000001)
        #         from numpy.linalg import det
        #         detCov = tf.linalg.det(
        #             tf.cast(tf.matmul(data, tf.transpose(data)), tf.float32) / tf.cast(num_samples, tf.float32))
        #         normalization = tf.math.pow(
        #             tf.cast((tf.math.multiply(2., tf.math.multiply(np.pi, tf.math.exp(1.0)))), tf.int32),
        #             num_dimensions)
        #         if detCov == 0:
        #             return -np.inf
        #         else:
        #             return 1 / 1 + (0.5 * log10(
        #                 epsilon + tf.math.multiply(tf.cast(normalization, tf.float32), tf.cast(detCov, tf.float32))))
        #
        #     rev = tf.map_fn(row_entropy, x, dtype=tf.float32)
        #     return rev
        #
        # import tensorflow as tf
        # k = topk
        # threshold = 10
        # p = (tf_a1 + tf.abs(tf_a1)) / 2
        #
        # values, indices = tf.nn.top_k(p, k/2)
        # my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)
        # my_range_repeated = tf.tile(my_range, [1, k/2])
        # full_indices = tf.stack([my_range_repeated, indices],
        #                         axis=2)
        # full_indices = tf.reshape(full_indices, [-1, 2])
        #
        # P_reset = tf.sparse_to_dense(full_indices, tf.shape(tf_a1), tf.reshape(values, [-1]), default_value=0.,
        #                              validate_indices=False)
        # ent_p = rev_entropy(P_reset)
        # ent_mask_p = tf.greater(ent_p, threshold)
        # p_good_rows = tf.where(ent_mask_p, P_reset, tf.zeros_like(P_reset))
        #
        # p_sum_not_in_goodrows = tf.reduce_sum(p - P_reset)
        # number_of_good_items = tf.where(p_good_rows)
        # enegy = p_sum_not_in_goodrows / tf.cast(tf.shape(number_of_good_items)[0] + 1, dtype=tf.float32)
        # energy_matrice = 6.3 * tf.scatter_nd(tf.cast(number_of_good_items, tf.int32),
        #                                enegy * tf.cast(tf.ones(shape=(tf.shape(number_of_good_items)[0])),tf.float32),
        #                                shape=tf.shape(P_reset))
        # result_p = p_good_rows + energy_matrice
        #
        # ####story for the negative weights
        #
        # n = (tf_a1 - tf.abs(tf_a1))/ 2
        # ent_n = rev_entropy(-n)
        # ent_mask_n = tf.greater(ent_n, threshold)
        # n_good_rows = tf.where(ent_mask_n, -n, tf.zeros_like(n))
        # values2, indices2 = tf.nn.top_k(n_good_rows, k-k/2)
        # my_range = tf.expand_dims(tf.range(0, tf.shape(indices2)[0]), 1)
        # my_range_repeated = tf.tile(my_range, [1, k-k/2])
        # full_indices2 = tf.stack([my_range_repeated, indices2], axis=2)
        # full_indices2 = tf.reshape(full_indices2, [-1, 2])
        # N_reset = tf.sparse_to_dense(full_indices2, tf.shape(tf_a1), tf.reshape(values2, [-1]), default_value=0.,
        #                              validate_indices=False)
        # n_sum_not_in_goodrows = tf.reduce_sum(-n - N_reset)
        # n_number_of_good_items = tf.where(n_good_rows)
        # enegy_n = n_sum_not_in_goodrows / tf.cast(tf.shape(n_number_of_good_items)[0] + 1, dtype=tf.float32)
        # energy_matrice_n = 6.3 * tf.scatter_nd(tf.cast(n_number_of_good_items, tf.int32),
        #                                  enegy_n * tf.cast(tf.ones(shape=(tf.shape(n_number_of_good_items)[0])),tf.float32),
        #                                  shape=tf.shape(N_reset))
        # result_n = n_good_rows + energy_matrice_n
        # res = result_p - result_n
        # return res

    def kSparse(self, x, topk):
        print 'run regular k-sparse'
        dim = int(x.get_shape()[1])
        if topk > dim:
            warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
            topk = dim

        k = dim - topk
        values, indices = tf.nn.top_k(-x, k) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]

        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]

        full_indices = tf.stack([my_range_repeated, indices], axis=2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])

        to_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)

        res = tf.add(x, to_reset)

        return res

class Dense_tied(Dense):
    """
    A fully connected layer with tied weights.
    """
    def __init__(self, units,
                 activation=None, use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 tied_to=None, **kwargs):
        self.tied_to = tied_to

        super(Dense_tied, self).__init__(units=units,
                 activation=activation, use_bias=use_bias,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                 activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                 **kwargs)

    def build(self, input_shape):
        super(Dense_tied, self).build(input_shape)  # be sure you call this somewhere!
        if self.kernel in self.trainable_weights:
            self.trainable_weights.remove(self.kernel)

    def call(self, x, mask=None):
        # Use tied weights
        self.kernel = K.transpose(self.tied_to.kernel)
        output = K.dot(x, self.kernel)
        if self.use_bias:
            output += self.bias
        return self.activation(output)

class CustomModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, custom_model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CustomModelCheckpoint, self).__init__()
        self.custom_model = custom_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('CustomModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        model = self.custom_model
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            model.save_weights(filepath, overwrite=True)
                        else:
                            model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    model.save_weights(filepath, overwrite=True)
                else:
                    model.save(filepath, overwrite=True)

class VisualWeights(Callback):
    def __init__(self, save_path, per_epoch=15):
        super(VisualWeights, self).__init__()
        self.per_epoch = per_epoch
        self.filename, self.ext = os.path.splitext(save_path)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        if epoch % self.per_epoch == 0:
            weights = self.model.get_weights()[0]
            # weights /= np.max(np.abs(weights))
            weights = unitmatrix(weights, axis=0) # normalize
            # weights[np.abs(weights) < 1e-2] = 0
            heatmap(weights.T, '%s_%s%s'%(self.filename, epoch, self.ext))
