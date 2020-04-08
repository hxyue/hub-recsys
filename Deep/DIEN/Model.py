from tensorflow.keras import layers
import tensorflow as tf


from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest


print(tf.__version__)
print(tf.keras.__version__)


# pylint: disable=protected-access
_concat = rnn_cell_impl._concat
_like_rnncell = rnn_cell_impl._like_rnncell
# pylint: enable=protected-access



class ActivationUnit(tf.keras.Model):
    def __init__(self):
        #super(ActivationUnit, self).__init__()
        self.act_dense_layer = layers.Dense(3, activation='relu', name='act_dense_layer')
        self.act_dense_out = layers.Dense(1, activation='sigmoid', name='act_dense_out')
        layers.LSTM(32)

    def call(self, inputs):
        '''
        在数据处理环节处理好，每次输入的len是不变的，但可以有不同len的输入,不足max_len的补0
        target_embed:[batch,1,embed_size]
        user_embed:[batch,max_len,embed_size]
        '''
        target_embed = inputs[0]
        user_embed = inputs[1]
        embed_size, max_len = user_embed.shape[2], user_embed.shape[1]

        target_embed = tf.tile(target_embed, [1, max_len])
        target_embed = tf.reshape(target_embed, [-1, max_len, embed_size])

        act_input = tf.concat([target_embed, target_embed - user_embed, user_embed], axis=2)
        act_input = self.act_dense_layer(act_input)
        act_out = self.act_dense_out(act_input)

        return act_out

layer = ActivationUnit()
layer([tf.ones([2, 3]), tf.ones([2, 3, 3])])
layer.summary()


def weighted_sum(weight, inputs):
    '''
    weight:[batch,max_len,1]
    inputs:[batch,max_len,embed_size]
    out:[batch,1]
    '''
    return tf.reduce_sum(weight * inputs, 1)


class DIN(tf.keras.Model):
    def __init__(self):
        super(DIN, self).__init__()
        self.activationUnit = ActivationUnit()
        self.dense_layer1 = layers.Dense(8, activation='relu', name='dense_layer1')
        self.dense_layer2 = layers.Dense(3, activation='relu', name='dense_layer2')
        self.dense_out = layers.Dense(1, activation='sigmoid', name='dense_out')

    def call(self, inputs):
        '''
        user_goods_embed:[batch,max_len,embed_size]
        user_shops_embed:[batch,max_len,embed_size]
        target_good_embed:[batch,embed_size]
        target_shop_embed:[batch,embed_size]
        target_other_embed:[batch,embed_size]
        '''
        assert (len(inputs) == 5)
        user_goods_embed = inputs[0]
        user_shops_embed = inputs[1]
        target_good_embed = inputs[2]
        target_shop_embed = inputs[3]
        target_other_embed = inputs[4]

        target_goods_weight = self.activationUnit([target_good_embed, user_goods_embed])
        sum_goods_embed = weighted_sum(target_goods_weight, user_goods_embed)
        target_shops_weight = self.activationUnit([target_shop_embed, user_shops_embed])
        sum_shops_embed = weighted_sum(target_shops_weight, user_shops_embed)

        out = tf.concat([sum_goods_embed, sum_shops_embed, target_good_embed, target_shop_embed, target_other_embed],
                        axis=1)
        print('dense input:', out.shape)
        out = self.dense_layer1(out)
        out = self.dense_layer2(out)
        out = self.dense_out(out)
        return out



class DIEN(tf.keras.Model):
    def __init__(self):
        super(DIEN, self).__init__()


def _transpose_batch_time(x):
  """Transpose the batch and time dimensions of a Tensor.

  Retains as much of the static shape information as possible.

  Args:
    x: A tensor of rank 2 or higher.

  Returns:
    x transposed along the first two dimensions.

  Raises:
    ValueError: if `x` is rank 1 or lower.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
    raise ValueError(
        "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
        (x, x_static_shape))
  x_rank = array_ops.rank(x)
  x_t = array_ops.transpose(
      x, array_ops.concat(
          ([1, 0], math_ops.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tensor_shape.TensorShape([
          x_static_shape[1].value, x_static_shape[0].value
      ]).concatenate(x_static_shape[2:]))
  return x_t

def dynamic_rnn(cell, inputs, att_scores=None, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
  1
  """Creates a recurrent neural network specified by RNNCell `cell`.

  Performs fully dynamic unrolling of `inputs`.

  Example:

  ```python
  # create a BasicRNNCell
  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

  # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

  # defining initial state
  initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

  # 'state' is a tensor of shape [batch_size, cell_state_size]
  outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                     initial_state=initial_state,
                                     dtype=tf.float32)
  ```

  ```python
  # create 2 LSTMCells
  rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

  # create a RNN cell composed sequentially of a number of RNNCells
  multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

  # 'outputs' is a tensor of shape [batch_size, max_time, 256]
  # 'state' is a N-tuple where N is the number of LSTMCells containing a
  # tf.contrib.rnn.LSTMStateTuple for each cell
  outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                     inputs=data,
                                     dtype=tf.float32)
  ```


  Args:
    cell: An instance of RNNCell.
    inputs: The RNN inputs.
      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such
        elements.
      If `time_major == True`, this must be a `Tensor` of shape:
        `[max_time, batch_size, ...]`, or a nested tuple of such
        elements.
      This may also be a (possibly nested) tuple of Tensors satisfying
      this property.  The first two dimensions must match across all the inputs,
      but otherwise the ranks and other shape components may differ.
      In this case, input to `cell` at each time-step will replicate the
      structure of these tuples, except for the time dimension (from which the
      time is taken).
      The input to `cell` at each time step will be a `Tensor` or (possibly
      nested) tuple of Tensors each with dimensions `[batch_size, ...]`.
    sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
      Used to copy-through state and zero-out outputs when past a batch
      element's sequence length.  So it's more for correctness than performance.
    initial_state: (optional) An initial state for the RNN.
      If `cell.state_size` is an integer, this must be
      a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
      If `cell.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors.
      If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
      If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
      Using `time_major = True` is a bit more efficient because it avoids
      transposes at the beginning and end of the RNN calculation.  However,
      most TensorFlow data is batch-major, so by default this function
      accepts input and emits output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to "rnn".

  Returns:
    A pair (outputs, state) where:

    outputs: The RNN output `Tensor`.

      If time_major == False (default), this will be a `Tensor` shaped:
        `[batch_size, max_time, cell.output_size]`.

      If time_major == True, this will be a `Tensor` shaped:
        `[max_time, batch_size, cell.output_size]`.

      Note, if `cell.output_size` is a (possibly nested) tuple of integers
      or `TensorShape` objects, then `outputs` will be a tuple having the
      same structure as `cell.output_size`, containing Tensors having shapes
      corresponding to the shape data in `cell.output_size`.

    state: The final state.  If `cell.state_size` is an int, this
      will be shaped `[batch_size, cell.state_size]`.  If it is a
      `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
      If it is a (possibly nested) tuple of ints or `TensorShape`, this will
      be a tuple having the corresponding shapes. If cells are `LSTMCells`
      `state` will be a tuple containing a `LSTMStateTuple` for each cell.

  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """
  if not _like_rnncell(cell):
    raise TypeError("cell must be an instance of RNNCell")

  # By default, time_major==False and inputs are batch-major: shaped
  #   [batch, time, depth]
  # For internal calculations, we transpose to [time, batch, depth]
  flat_input = nest.flatten(inputs)

  if not time_major:
    # (B,T,D) => (T,B,D)
    flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
    flat_input = tuple(_transpose_batch_time(input_) for input_ in flat_input)

  parallel_iterations = parallel_iterations or 32
  if sequence_length is not None:
    sequence_length = math_ops.to_int32(sequence_length)
    if sequence_length.get_shape().ndims not in (None, 1):
      raise ValueError(
          "sequence_length must be a vector of length batch_size, "
          "but saw shape: %s" % sequence_length.get_shape())
    sequence_length = array_ops.identity(  # Just to find it in the graph.
        sequence_length, name="sequence_length")

  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with vs.variable_scope(scope or "rnn") as varscope:
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)
    batch_size = _best_effort_input_batch_size(flat_input)

    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If there is no initial_state, you must give a dtype.")
      state = cell.zero_state(batch_size, dtype)

    def _assert_has_shape(x, shape):
      x_shape = array_ops.shape(x)
      packed_shape = array_ops.stack(shape)
      return control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
          ["Expected shape for Tensor %s is " % x.name,
           packed_shape, " but saw shape: ", x_shape])

    if sequence_length is not None:
      # Perform some shape validation
      with ops.control_dependencies(
          [_assert_has_shape(sequence_length, [batch_size])]):
        sequence_length = array_ops.identity(
            sequence_length, name="CheckSeqLen")

    inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

    (outputs, final_state) = _dynamic_rnn_loop(
        cell,
        inputs,
        state,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        att_scores = att_scores,
        sequence_length=sequence_length,
        dtype=dtype)

    # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    # If we are performing batch-major calculations, transpose output back
    # to shape [batch, time, depth]
    if not time_major:
      # (T,B,D) => (B,T,D)
      outputs = nest.map_structure(_transpose_batch_time, outputs)

    return (outputs, final_state)





class Model_DIN_V2_Gru_Vec_attGru_Neg(Model):

    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DIN_V2_Gru_Vec_attGru_Neg, self).__init__(n_uid, n_mid, n_cat,
                                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                          use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                         self.noclk_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)


layer = DIN()
layer([tf.ones([2, 3, 10]), tf.ones([2, 3, 10]), tf.ones([2, 10]), tf.ones([2, 10]), tf.ones([2, 10])])
layer.summary(line_length=100)