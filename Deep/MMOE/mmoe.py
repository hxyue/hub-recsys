#coding:utf-8
import tensorflow as tf
from tensorflow import tensordot, expand_dims
from tensorflow.keras import layers, Model, initializers, regularizers, activations, constraints, Input


from tensorflow.keras.backend import expand_dims,repeat_elements,sum

class MMoE(layers.Layer):
    """
    Multi-gate Mixture-of-Experts model.
    """

    def __init__(self,
                 units,
                 num_experts,
                 num_tasks,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',
                 expert_bias_initializer='zeros',
                 gate_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """
         Method for instantiating MMoE layer.
        :param units: Number of hidden units
        :param num_experts: Number of experts
        :param num_tasks: Number of tasks
        :param use_expert_bias: Boolean to indicate the usage of bias in the expert weights
        :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights
        :param expert_activation: Activation function of the expert weights
        :param gate_activation: Activation function of the gate weights
        :param expert_bias_initializer: Initializer for the expert bias
        :param gate_bias_initializer: Initializer for the gate bias
        :param expert_bias_regularizer: Regularizer for the expert bias
        :param gate_bias_regularizer: Regularizer for the gate bias
        :param expert_bias_constraint: Constraint for the expert bias
        :param gate_bias_constraint: Constraint for the gate bias
        :param expert_kernel_initializer: Initializer for the expert weights
        :param gate_kernel_initializer: Initializer for the gate weights
        :param expert_kernel_regularizer: Regularizer for the expert weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param expert_kernel_constraint: Constraint for the expert weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class
        """
        super(MMoE, self).__init__(**kwargs)

        # Hidden nodes parameter
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

        # Activation parameter
        #self.expert_activation = activations.get(expert_activation)
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []
        for i in range(self.num_experts):
            self.expert_layers.append(layers.Dense(self.units, activation=self.expert_activation,
                                                   use_bias=self.use_expert_bias,
                                                   kernel_initializer=self.expert_kernel_initializer,
                                                   bias_initializer=self.expert_bias_initializer,
                                                   kernel_regularizer=self.expert_kernel_regularizer,
                                                   bias_regularizer=self.expert_bias_regularizer,
                                                   activity_regularizer=None,
                                                   kernel_constraint=self.expert_kernel_constraint,
                                                   bias_constraint=self.expert_bias_constraint))
        for i in range(self.num_tasks):
            self.gate_layers.append(layers.Dense(self.num_experts, activation=self.gate_activation,
                                                 use_bias=self.use_gate_bias,
                                                 kernel_initializer=self.gate_kernel_initializer,
                                                 bias_initializer=self.gate_bias_initializer,
                                                 kernel_regularizer=self.gate_kernel_regularizer,
                                                 bias_regularizer=self.gate_bias_regularizer, activity_regularizer=None,
                                                 kernel_constraint=self.gate_kernel_constraint,
                                                 bias_constraint=self.gate_bias_constraint))

    def call(self, inputs):
        """
        Method for the forward function of the layer.
        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        #assert input_shape is not None and len(input_shape) >= 2

        expert_outputs, gate_outputs, final_outputs = [], [], []
        for expert_layer in self.expert_layers:
            expert_output = expand_dims(expert_layer(inputs), axis=2)
            expert_outputs.append(expert_output)
        expert_outputs = tf.concat(expert_outputs,2)

        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(inputs))

        for gate_output in gate_outputs:
            expanded_gate_output = expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_outputs * repeat_elements(expanded_gate_output, self.units, axis=1)
            final_outputs.append(sum(weighted_expert_output, axis=2))
        # 返回的矩阵维度 num_tasks * batch * units
        return final_outputs


mmoe = MMoE(units=16,num_experts=8,num_tasks=2)
mmoe(tf.zeros((4, 1)))


