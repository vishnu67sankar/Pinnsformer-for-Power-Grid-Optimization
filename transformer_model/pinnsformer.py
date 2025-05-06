import warnings 
import os
import pathlib
import json
import warnings
import numpy as np
import tensorflow as tf
import scipy.sparse 
from tqdm import tqdm 
import datetime
from lips.benchmark.powergridBenchmark import PowerGridBenchmark, get_env, get_kwargs_simulator_scenario
from lips.dataset.scaler.powergrid_scaler import PowerGridScaler 
from lips.utils import NpEncoder


def KKTP7ABb(obs):
    """TensorFlow version of KKTP7ABb."""
    sub_number = 118 
    line_number = 186 

    line_or_to_subid = tf.constant(obs.line_or_to_subid, dtype=tf.int32)
    line_ex_to_subid = tf.constant(obs.line_ex_to_subid, dtype=tf.int32)

    indices_B1 = tf.stack([tf.range(line_number), line_or_to_subid], axis=1)
    updates_B1 = tf.ones(line_number, dtype=tf.float32)
    B1 = tf.scatter_nd(indices_B1, updates_B1, shape=[line_number, sub_number])

    indices_B2 = tf.stack([tf.range(line_number), line_ex_to_subid], axis=1)
    updates_B2 = tf.ones(line_number, dtype=tf.float32)
    B2 = tf.scatter_nd(indices_B2, updates_B2, shape=[line_number, sub_number])

    B1 = tf.transpose(B1) 
    B2 = tf.transpose(B2) 

    A = tf.eye(sub_number, dtype=tf.float32)
    B = -tf.concat((B1, B2), axis=1) 
    b = tf.zeros(sub_number, dtype=tf.float32)

    
    BBT = tf.matmul(B, tf.transpose(B)) # calculate inverse: B @ B.T
    BBT_inv = tf.linalg.inv(BBT + tf.eye(tf.shape(BBT)[0], dtype=tf.float32)*1e-6) 

    A_star = -tf.matmul(tf.transpose(B), tf.matmul(BBT_inv, A))
    B_star = tf.eye(line_number*2, dtype=tf.float32) - tf.matmul(tf.transpose(B), tf.matmul(BBT_inv, B)) 
    b_star = tf.matmul(tf.transpose(B), tf.expand_dims(tf.matmul(BBT_inv, tf.expand_dims(b, axis=-1))[:,0], axis=-1)) 

    return A_star, B_star, b_star

def KKTP7Xp(obs, data_inversed):
    """TensorFlow version of KKTP7Xp."""
    
    batch_size = tf.shape(data_inversed)[0]
    sub_number = 118 
    
    prod_p_data = data_inversed[:, :62]
    load_p_data = data_inversed[:, 124:223] # example slice

    gen_to_subid = tf.constant(obs.gen_to_subid, dtype=tf.int32)
    load_to_subid = tf.constant(obs.load_to_subid, dtype=tf.int32) 

    
    gen_to_subid_one_hot = tf.cast(tf.equal(tf.expand_dims(gen_to_subid, 0), tf.expand_dims(tf.range(sub_number), 1)), tf.float32) 
    gen_power_sums = tf.matmul(gen_to_subid_one_hot, tf.transpose(tf.cast(prod_p_data, tf.float32))) 

    load_to_subid_one_hot = tf.cast(tf.equal(tf.expand_dims(load_to_subid, 0), tf.expand_dims(tf.range(sub_number), 1)), tf.float32) 
    load_power_sums = tf.matmul(load_to_subid_one_hot, tf.transpose(tf.cast(load_p_data, tf.float32)))

    X_p = gen_power_sums - load_power_sums
    return X_p

def KKTP4(data_tau, output_inversed):
    """ Zeros out predictions for disconnected lines. """
    line_status = data_tau[:, 0:186] 
    is_disconnected = tf.cast(line_status, dtype=tf.bool) # 1 means disconnected

    
    mask = tf.logical_not(is_disconnected) # true where connected, false where disconnected
    mask_float = tf.cast(mask, dtype=output_inversed.dtype)

    output_a_or = output_inversed[:, :186]*mask_float
    output_a_ex = output_inversed[:, 186:372]*mask_float
    output_p_or = output_inversed[:, 372:558]*mask_float
    output_p_ex = output_inversed[:, 558:744]*mask_float
    output_v_or = output_inversed[:, 744:930]*mask_float
    output_v_ex = output_inversed[:, 930:1116]*mask_float
    
    output_modified = tf.concat([output_a_or, output_a_ex, output_p_or, output_p_ex, output_v_or, output_v_ex], axis=1)
    
    return output_modified


def calculate_P8(a_or_pred, a_ex_pred, p_or_pred, p_ex_pred, obs, 
                      YBus_edge_indices, RBus_edge):
    """
    Calculates the P8 physics constraint loss term in TensorFlow.

    P8 = | mean( (p_or + p_ex)*scale - ( (a_or+a_ex)/2 )^2*R_line ) |

    Args:
        a_or_pred (tf.Tensor): Predicted current at origin, shape [batch, 186].
        a_ex_pred (tf.Tensor): Predicted current at extremity, shape [batch, 186].
        p_or_pred (tf.Tensor): Predicted power at origin, shape [batch, 186].
        p_ex_pred (tf.Tensor): Predicted power at extremity, shape [batch, 186].
        obs (object): Grid observation object containing line_or_to_subid, etc.
        YBus_edge_indices (tf.Tensor): YBus edge indices tensor, 
                                           shape [batch, 2, max_edges], dtype=int32.
                                           Contains (from_node, to_node) pairs.
        RBus_edge (tf.Tensor): Pre-calculated line resistances corresponding 
                                  to YBus edges, shape [batch, max_edges], dtype=float32.

    Returns:
        tf.Tensor: Scalar tensor representing the P8 loss value.
    """
    batch_size = tf.shape(a_or_pred)[0]
    num_lines = tf.shape(a_or_pred)[1] 
    max_edges = tf.shape(YBus_edge_indices)[2]

    line_or_to_subid = tf.constant(obs.line_or_to_subid, dtype=tf.int32) 
    line_ex_to_subid = tf.constant(obs.line_ex_to_subid, dtype=tf.int32) 
    line_s_to_l_sub = tf.stack([line_or_to_subid, line_ex_to_subid], axis=0) 
    comparison = line_s_to_l_sub[0] < line_s_to_l_sub[1]
    line_s_to_l_sub_0 = tf.where(comparison, line_s_to_l_sub[0], line_s_to_l_sub[1])
    line_s_to_l_sub_1 = tf.where(comparison, line_s_to_l_sub[1], line_s_to_l_sub[0])
    line_s_to_l_sub_sorted = tf.cast(tf.stack([line_s_to_l_sub_0, line_s_to_l_sub_1], axis=0), dtype=tf.int32) 

    ybus_t = tf.transpose(YBus_edge_indices, perm=[0, 2, 1]) 
    # lines_t shape: [186, 2]
    lines_t = tf.transpose(line_s_to_l_sub_sorted, perm=[1, 0]) 
    ybus_t_exp = tf.expand_dims(ybus_t, axis=2) 
    lines_t_exp = tf.expand_dims(tf.expand_dims(lines_t, axis=0), axis=0) 
    matches = tf.reduce_all(tf.equal(ybus_t_exp, lines_t_exp), axis=3) 

    matches_int = tf.cast(matches, dtype=tf.int64)
    result_indices_per_line = tf.argmax(matches_int, axis=1) 

    any_match_per_line = tf.reduce_any(matches, axis=1) 
    no_match_mask = tf.logical_and(tf.equal(result_indices_per_line, 0), 
                                   tf.logical_not(any_match_per_line))

    result_indices = tf.where(no_match_mask, 
                              tf.constant(-1, dtype=tf.int64), 
                              result_indices_per_line)
    

    batch_indices = tf.range(batch_size, dtype=tf.int64)
    batch_indices_exp = tf.expand_dims(batch_indices, axis=1) 
    batch_indices_tiled = tf.tile(batch_indices_exp, [1, num_lines])
    indices_for_gather = tf.stack([batch_indices_tiled, tf.maximum(result_indices, 0)], axis=-1) 
    RBus_gathered = tf.gather_nd(RBus_edge, indices_for_gather) 

    RBus_selected = tf.where(tf.equal(result_indices, -1), tf.zeros_like(RBus_gathered), RBus_gathered) 
    a_or_ex_pred_avg = (a_or_pred + a_ex_pred)*0.5

    P81 = (p_or_pred + p_ex_pred)*tf.constant(17.498, dtype=tf.float32)
    P82 = tf.multiply(tf.pow(a_or_ex_pred_avg, 2), RBus_selected)

    P8 = tf.abs(tf.reduce_mean(P81 - P82)) # most useless effort, not gonna implement it

    return P8

def calculate_physics_loss(predict, target, data_full_inversed, YBus_edge_indices, RBus_edge, obs):
    """
    Calculates physics-based constraints P1-P8 and MSE loss using TensorFlow.
    
    Args:
        predict (tf.Tensor): Final (KKT adjusted) predicted values (shape [batch, 1116]).
        target (tf.Tensor): Target values (unscaled) (shape [batch, 1116]).
        data_full_inversed (tf.Tensor): Input data (unscaled, including tau) (shape [batch, features]).
        YBus_edge_indices (tf.Tensor): YBus edge indices (shape [batch, 2, edges], dtype=tf.int32).
        RBus_edge (tf.Tensor): Calculated resistance tensor (shape [batch, edges]).
        obs (object): Observation object containing grid info (assumed accessible).

    Returns:
        loss (tf.Tensor): The combined loss value.
        loss_components (dict): Dictionary containing individual loss components.
    """
    batch_size = tf.shape(predict)[0]
    a_or_pred = predict[:,:186] 
    a_ex_pred = predict[:,186:372]
    p_or_pred = predict[:,372:558]
    p_ex_pred = predict[:,558:744]
    v_or_pred = predict[:,744:930]
    v_ex_pred = predict[:,930:]

    
    prod_p_data = data_full_inversed[:,:62] 
    load_p_data = data_full_inversed[:,124:223] 
    
    line_status_data = data_full_inversed[:,322:508] 

    
    condition_1 = tf.logical_or(a_or_pred < 0, a_ex_pred < 0) # P1 Constraint
    P1 = tf.reduce_mean(tf.cast(condition_1, tf.float32))

    
    condition_2 = tf.logical_or(v_or_pred < 0, v_ex_pred < 0) # P2 Constraint
    P2 = tf.reduce_mean(tf.cast(condition_2, tf.float32))

    p_or_ex_pred_sum = p_or_pred + p_ex_pred # P3 Constraint
    condition_3 = p_or_ex_pred_sum < 0
    P3 = tf.reduce_mean(tf.cast(condition_3, tf.float32))

    
    # P4 Constraint 
    # P4 is handled by KKTP4 before this loss function is called. which penalized non-zero values *after* zeroing them out (effectively always 0 if KKTP4 is applied first).
    # if KKTP4 is applied, this P4 calculation becomes redundant. 
    # if KKTP4 is NOT applied, this calculates the original P4 penalty.
    disconnect_line_indices = tf.where(line_status_data == 1) 
    a_or_pred_disconnected = tf.gather_nd(a_or_pred, disconnect_line_indices)
    a_ex_pred_disconnected = tf.gather_nd(a_ex_pred, disconnect_line_indices)
    p_or_pred_disconnected = tf.gather_nd(p_or_pred, disconnect_line_indices)
    p_ex_pred_disconnected = tf.gather_nd(p_ex_pred, disconnect_line_indices)
    a_or_ex_pred_abs_sum = tf.abs(a_or_pred_disconnected) + tf.abs(a_ex_pred_disconnected)
    p_or_ex_pred_abs_sum = tf.abs(p_or_pred_disconnected) + tf.abs(p_ex_pred_disconnected)
    P4 = tf.cond(tf.shape(disconnect_line_indices)[0] > 0, lambda: tf.reduce_mean(a_or_ex_pred_abs_sum + p_or_ex_pred_abs_sum), lambda: tf.constant(0.0, dtype=tf.float32))
    P4 = tf.cast(P4, tf.float32)


    
    p_or_ex_pred_dim1sum = tf.reduce_sum(p_ex_pred + p_or_pred, axis=1)  #  P5 Constraint 
    prod_p_data_dim1sum = tf.reduce_sum(prod_p_data, axis=1) 
    safe_prod_p_data_dim1sum = tf.where(prod_p_data_dim1sum == 0, tf.ones_like(prod_p_data_dim1sum), prod_p_data_dim1sum)
    energy_loss_ratio = p_or_ex_pred_dim1sum / safe_prod_p_data_dim1sum
    penalty1 = tf.where(energy_loss_ratio > 0.04, 200.0*(energy_loss_ratio - 0.04), 0.0)
    condition_5 = tf.logical_and(energy_loss_ratio < 0.005, energy_loss_ratio > 0)
    penalty2 = tf.where(condition_5, 200.0*(0.005 - energy_loss_ratio), 0.0)
    penalty3 = tf.where(energy_loss_ratio < 0, 500.0*(0.005 - energy_loss_ratio), 0.0) 
    total_energy_penalty = penalty1 + penalty2 + penalty3
    P5 = tf.reduce_mean(total_energy_penalty)

    
    line_or_to_subid = tf.constant(obs.line_or_to_subid, dtype=tf.int32)  
    line_ex_to_subid = tf.constant(obs.line_ex_to_subid, dtype=tf.int32) 
    line_s_to_l_sub = tf.stack([line_or_to_subid, line_ex_to_subid], axis=0)
    comparison = line_s_to_l_sub[0] < line_s_to_l_sub[1]
    line_s_to_l_sub_0 = tf.where(comparison, line_s_to_l_sub[0], line_s_to_l_sub[1])
    line_s_to_l_sub_1 = tf.where(comparison, line_s_to_l_sub[1], line_s_to_l_sub[0])
    line_s_to_l_sub = tf.cast(tf.stack([line_s_to_l_sub_0, line_s_to_l_sub_1], axis=0), dtype=tf.int32) 

    a_or_ex_pred_avg = (a_or_pred + a_ex_pred)*0.5 

    P6 = tf.constant(0.0, dtype=tf.float32) #  P6 and P7 are handled by KKT step before loss 
    P7 = tf.constant(0.0, dtype=tf.float32) # loss penalty is 0, effect comes from KKT step
    P8 = tf.constant(0.0, dtype=tf.float32)
    MSE = tf.reduce_mean(tf.pow((predict - target), 2))


    total_loss = 10*MSE + P1 + P2 + P3 + P4 + P5 + P6 + P7 + 0.0*P8 

    loss_components = {'MSE': MSE, 'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4, 'P5': P5, 'P6': P6, 'P7': P7, 'P8': P8, 'TotalLoss': total_loss}
    return total_loss, loss_components


def scaled_dot_product_attention(q, k, v, mask):
  matmul_qk = tf.matmul(q, k, transpose_b=True)
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  if mask is not None:
    scaled_attention_logits += (mask*-1e9)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
  output = tf.matmul(attention_weights, v)
  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    assert d_model % self.num_heads == 0
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x):
    batch_size = tf.shape(x)[0]
    depth = self.d_model//self.num_heads
    x = tf.reshape(x, (batch_size, -1, self.num_heads, depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask=None, training=False):
    batch_size = tf.shape(q)[0]
    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
    output = self.dense(concat_attention)
    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation=tf.keras.activations.gelu), tf.keras.layers.Dense(d_model)])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
   
  def call(self, x, training=False, mask=None):
    attn_output, attention_weights = self.mha(x, x, x, mask, training=training)
  
    out1 = self.layernorm1(x + attn_output)
    ffn_output = self.ffn(out1)
   
    out2 = self.layernorm2(out1 + ffn_output)
    return out2, attention_weights
   
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(CrossAttention, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads=2) 
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6) 


    def call(self, q, kv, training=False):
        x, _ = self.mha(kv, kv, q, training=training)
        ffn_output = self.ffn(x)
        out2 = self.layernorm(q + ffn_output) 
        return out2

class TransformerEncoderTrans(tf.keras.layers.Layer): 
  def __init__(self, output_dim, x_attr_sizes, t_attr_sizes, y_attr_sizes,
               num_layers, d_model, num_heads, dff):
    super(TransformerEncoderTrans, self).__init__()
    self.attr_sizes = list(x_attr_sizes) + list(t_attr_sizes)
    self.seq_len = len(y_attr_sizes)
    self.y_attr_sizes = y_attr_sizes 
    self.latent_emb = tf.keras.layers.Embedding(len(y_attr_sizes), d_model)
    self.cross = CrossAttention(d_model, dff)
    self.embeddings = [tf.keras.layers.Dense(d_model, activation='elu')
                       for _ in self.attr_sizes]
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.d_model = d_model
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff)
                       for _ in range(num_layers)]
    self.dec_emb_layer = point_wise_feed_forward_network(d_model, dff) 
    self.dec_layers = [tf.keras.layers.Dense(s) for s in y_attr_sizes] 

  def call(self, inp):
    split = tf.split(inp, self.attr_sizes, axis=1)
    x = [tf.expand_dims(lay(s), axis=1) for (lay, s) in zip(self.embeddings, split)]
    x = tf.concat(x, axis=1) 
    x = self.ffn(x)
    
    lat = self.latent_emb(tf.range(self.seq_len)) 
    lat = tf.expand_dims(lat, 0)
    lat = tf.tile(lat, [tf.shape(inp)[0], 1, 1]) 

    x = self.cross(lat, x) 

    for lay in self.enc_layers:
      x, _ = lay(x)

    x = self.dec_emb_layer(x) 
    

    outputs = []
    for i, head_layer in enumerate(self.dec_layers):
        outputs.append(head_layer(x[:, i, :]))
        
    final_output = tf.concat(outputs, axis=-1)
    return final_output


class ProcessMeanStd(tf.keras.layers.Layer):
    def __init__(self, m_x_list, sd_x_list, m_y_list, sd_y_list):
        """
        Initializes the unscaler, handling potential scalar means/stds from PowerGridScaler.

        Args:
            m_x_list (list): List of numpy arrays/scalars for input means.
            sd_x_list (list): List of numpy arrays/scalars for input std devs.
            m_y_list (list): List of numpy arrays/scalars for output means.
            sd_y_list (list): List of numpy arrays/scalars for output std devs.
        """
        super().__init__()

        def process_list(mean_list, std_list):
            """ Helper function to process mean and std dev lists, expanding scalars. """
            if not mean_list or not std_list:
                print(f"Warning: Empty mean or std list provided to ProcessMeanStd.")
                return None, None
                
            processed_means = []
            processed_stds = []
            
            for i, (m, sd) in enumerate(zip(mean_list, std_list)):
                
                current_mean = np.array(m, dtype=np.float32) 
                current_std = np.array(sd, dtype=np.float32)  
                
                target_len = -1
                if current_std.ndim == 1:
                    target_len = len(current_std)
                elif current_mean.ndim == 1: 
                    target_len = len(current_mean)
                elif current_mean.ndim == 0 and current_std.ndim == 0: 
                    target_len = 1 
                
                if current_mean.ndim == 0:
                    expanded_mean = np.full(target_len, current_mean.item(), dtype=np.float32)
                    processed_means.append(expanded_mean)
                elif current_mean.ndim == 1:
                    if len(current_mean) != target_len:
                        raise ValueError(f"Mismatch in length between determined target ({target_len}) and mean ({len(current_mean)}) at index {i}")
                    processed_means.append(current_mean) 
               
                if current_std.ndim == 0: 
                    expanded_std = np.full(target_len, current_std.item(), dtype=np.float32)
                    processed_stds.append(expanded_std)
                elif current_std.ndim == 1:
                    if len(current_std) != target_len:
                        raise ValueError(f"Mismatch in length between determined target ({target_len}) and std dev ({len(current_std)}) at index {i}")
                    processed_stds.append(current_std)
                
            if not processed_means or not processed_stds:
                print("Warning: Processed mean/std lists are empty after processing in ProcessMeanStd.")
                return None, None
                
            try:
                final_mean_concat = np.concatenate(processed_means, axis=0)
                final_std_concat = np.concatenate(processed_stds, axis=0)
            except ValueError as e:
                print("Error during concatenation in ProcessMeanStd:")
                print(f"Processed Mean shapes: {[p.shape for p in processed_means]}")
                print(f"Processed Std shapes: {[p.shape for p in processed_stds]}")
                raise e

            final_mean_tensor = tf.constant(final_mean_concat, dtype=tf.float32)
            final_std_tensor = tf.constant(final_std_concat, dtype=tf.float32)
            return final_mean_tensor, final_std_tensor

        self.m_x, self.sd_x = process_list(m_x_list, sd_x_list)
        self.m_y, self.sd_y = process_list(m_y_list, sd_y_list)


    def call(self, x_scaled, y_scaled, tau):
        if self.m_x is not None and self.sd_x is not None:
            x_unscaled = x_scaled*(self.sd_x + 1e-7) + self.m_x
        else:
            x_unscaled = x_scaled 

        if self.m_y is not None and self.sd_y is not None:
            y_unscaled = y_scaled*(self.sd_y + 1e-7) + self.m_y
        else:
             y_unscaled = y_scaled 

        return x_unscaled, y_unscaled, tau


class PinnformerModel(tf.keras.Model):

    def __init__(self, transformer_core, unscaler, obs, x_feature_dim, **kwargs): 
        super().__init__(**kwargs)
        self.transformer_core = transformer_core
        self.unscaler = unscaler 
        self.obs = obs 
        self.input_feature_dim = x_feature_dim 
       
        self.P7A_star, self.P7B_star, self.P7b_star = KKTP7ABb(obs)
        
        self.loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse")
        self.p1_tracker = tf.keras.metrics.Mean(name="p1")
        self.p2_tracker = tf.keras.metrics.Mean(name="p2")
        self.p3_tracker = tf.keras.metrics.Mean(name="p3")
        self.p4_tracker = tf.keras.metrics.Mean(name="p4")
        self.p5_tracker = tf.keras.metrics.Mean(name="p5")
        self.p8_tracker = tf.keras.metrics.Mean(name="p8")

    def call(self, inputs, training=False):
        return self.transformer_core(inputs, training=training)

    @property
    def metrics(self):
        return [
            self.loss_tracker, self.mse_tracker, self.p1_tracker, self.p2_tracker,
            self.p3_tracker, self.p4_tracker, self.p5_tracker, self.p8_tracker
        ]

    def train_step(self, batch_data):
        inputs_scaled, ybus_dict, targets_scaled = batch_data 
        YBus_edge_indices = ybus_dict['edge_indices'] 
        RBus_edge = ybus_dict['RBus_edge']
        input_features_scaled = inputs_scaled[:, :self.input_feature_dim] 
        tau_features = inputs_scaled[:, self.input_feature_dim:] 
        
        with tf.GradientTape() as tape:
            predictions_scaled = self.transformer_core(inputs_scaled, training=True)
            input_features_unscaled, predictions_unscaled, tau_unscaled = self.unscaler(input_features_scaled, predictions_scaled, tau_features)
            
            _, targets_unscaled, _ = self.unscaler(input_features_scaled, targets_scaled, tau_features )
            data_full_inversed = tf.concat([input_features_unscaled, tau_unscaled], axis=1)

            P7X = KKTP7Xp(self.obs, data_full_inversed) 
            p_tmp = tf.transpose(predictions_unscaled[:, 372:744]) 
            kkt_term1 = tf.matmul(self.P7A_star, P7X) 
            kkt_term2 = tf.matmul(self.P7B_star, p_tmp) 
            kkt_update = kkt_term1 + kkt_term2 + self.P7b_star 
            predictions_p_updated = tf.transpose(kkt_update) 
            final_predictions_before_p4 = tf.concat([
                predictions_unscaled[:, :372],
                predictions_p_updated,
                predictions_unscaled[:, 744:]], axis=1)

            final_predictions = KKTP4(tau_unscaled, final_predictions_before_p4) 

            total_loss, loss_components = calculate_physics_loss(
                predict=final_predictions, 
                target=targets_unscaled,    
                data_full_inversed=data_full_inversed, 
                YBus_edge_indices=YBus_edge_indices, 
                RBus_edge=RBus_edge,                
                obs=self.obs)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm=1.0) 
        self.optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss_components['TotalLoss'])
        self.mse_tracker.update_state(loss_components['MSE'])
        self.p1_tracker.update_state(loss_components['P1'])
        self.p2_tracker.update_state(loss_components['P2'])
        self.p3_tracker.update_state(loss_components['P3'])
        self.p4_tracker.update_state(loss_components['P4'])
        self.p5_tracker.update_state(loss_components['P5'])
        self.p8_tracker.update_state(loss_components['P8'])
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, batch_data):
        inputs_scaled, ybus_dict, targets_scaled = batch_data 
        YBus_edge_indices = ybus_dict['edge_indices'] 
        RBus_edge = ybus_dict['RBus_edge']
        input_features_scaled = inputs_scaled[:, :self.input_feature_dim]
        tau_features = inputs_scaled[:, self.input_feature_dim:] 

        predictions_scaled = self.transformer_core(inputs_scaled, training=False)

        input_features_unscaled, predictions_unscaled, tau_unscaled = self.unscaler(input_features_scaled, predictions_scaled, tau_features)
        _, targets_unscaled, _ = self.unscaler(input_features_scaled, targets_scaled, tau_features)
        data_full_inversed = tf.concat([input_features_unscaled, tau_unscaled], axis=1)

        P7X = KKTP7Xp(self.obs, data_full_inversed) 
        p_tmp = tf.transpose(predictions_unscaled[:, 372:744]) 
        kkt_term1 = tf.matmul(self.P7A_star, P7X) 
        kkt_term2 = tf.matmul(self.P7B_star, p_tmp) 
        kkt_update = kkt_term1 + kkt_term2 + self.P7b_star 
        predictions_p_updated = tf.transpose(kkt_update) 
        final_predictions = tf.concat([
            predictions_unscaled[:, :372], predictions_p_updated, predictions_unscaled[:, 744:]
        ], axis=1)

        final_predictions = KKTP4(tau_unscaled, final_predictions)
        total_loss, loss_components = calculate_physics_loss(
            predict=final_predictions,  
            target=targets_unscaled,    
            data_full_inversed=data_full_inversed, 
            YBus_edge_indices=YBus_edge_indices, 
            RBus_edge=RBus_edge,              
            obs=self.obs)

       
        self.loss_tracker.update_state(loss_components['TotalLoss'])
        self.mse_tracker.update_state(loss_components['MSE'])
        self.p1_tracker.update_state(loss_components['P1'])
        self.p2_tracker.update_state(loss_components['P2'])
        self.p3_tracker.update_state(loss_components['P3'])
        self.p4_tracker.update_state(loss_components['P4'])
        self.p5_tracker.update_state(loss_components['P5'])
        self.p8_tracker.update_state(loss_components['P8'])

        return {m.name: m.result() for m in self.metrics}

def process_ybus(ybus_csr_matrix):
    """
    Processes a single stacked sparse YBus CSR matrix into dense TF tensors 
    for edge lists and calculates Rbus_edge.

    Args:
        ybus_csr_matrix (scipy.sparse.csr_matrix): Input YBus data as a single
            CSR matrix shape (num_samples, N*N), where N is the number of nodes.

    Returns:
        tuple: (YBus_edge_indices, RBus_edge)
               - YBus_edge_indices: tf.Tensor shape [batch, 2, max_edges], dtype=tf.int32
               - RBus_edge: tf.Tensor shape [batch, max_edges], dtype=tf.float32
    """
    if not isinstance(ybus_csr_matrix, scipy.sparse.csr_matrix):
        if ybus_csr_matrix is None:
            print("Warning: YBus data is None. Cannot process.")
            return tf.zeros((0, 2, 0), dtype=tf.int32), tf.zeros((0, 0), dtype=tf.float32) 
        else:
            raise TypeError("Expected ybus_csr_matrix to be a SciPy CSR sparse matrix, "f"but got {type(ybus_csr_matrix)}")

    num_samples = ybus_csr_matrix.shape[0]
    flattened_size = ybus_csr_matrix.shape[1]
    num_nodes = int(np.sqrt(flattened_size))
    
    print(f"Processing YBus: {num_samples} samples, {num_nodes} nodes per sample.")
    all_edges_list = []
    max_num_edges = 0
    
    print("Extracting edges from each sample row...")
    for i in tqdm(range(num_samples)):
        sample_sparse_row = ybus_csr_matrix.getrow(i)
        sample_sparse_matrix = sample_sparse_row.reshape((num_nodes, num_nodes))
        coo = sample_sparse_matrix.tocoo()

        edges = np.vstack((coo.row, coo.col, coo.data.astype(np.complex64))) 
        
        all_edges_list.append(edges)
        max_num_edges = max(max_num_edges, edges.shape[1])

    num_valid_samples = len(all_edges_list) 

    print(f"Padding edge lists to max_num_edges: {max_num_edges}")
    padded_ybus_indices_np = np.zeros((num_valid_samples, 2, max_num_edges), dtype=np.int32)
    padded_ybus_values_np = np.zeros((num_valid_samples, max_num_edges), dtype=np.complex64) 

    for i in tqdm(range(num_valid_samples)):
        edges_sample = all_edges_list[i]
        num_edges_in_sample = edges_sample.shape[1]
        padded_ybus_indices_np[i, :, :num_edges_in_sample] = edges_sample[0:2, :]
        padded_ybus_values_np[i, :num_edges_in_sample] = edges_sample[2, :]

    YBus_edge_indices = tf.constant(padded_ybus_indices_np, dtype=tf.int32)
    YBus_edge_values = tf.constant(padded_ybus_values_np, dtype=tf.complex64)

    print("Calculating RBus_edge...")
    from_nodes = YBus_edge_indices[:, 0, :] 
    to_nodes = YBus_edge_indices[:, 1, :]   

    is_diagonal = tf.equal(from_nodes, to_nodes)
    ybus_handled = tf.where(is_diagonal, YBus_edge_values, -YBus_edge_values) 

    real_ybus = tf.math.real(ybus_handled)
   
    safe_real_ybus = tf.where(tf.abs(real_ybus) < 1e-9, tf.ones_like(real_ybus)*1e-9, real_ybus)
    
    
    RBus_edge = tf.where(is_diagonal, tf.zeros_like(real_ybus, dtype=tf.float32), tf.cast(1.0 / safe_real_ybus, dtype=tf.float32))

    print("YBus processing complete.")
    return YBus_edge_indices, RBus_edge


def create_dataset(lips_dataset, scaler, YBus_indices, RBus_edges, batch_size, shuffle=False):
    """
    Creates a tf.data.Dataset from LIPS dataset components, using PowerGridScaler correctly.
    Includes debugging for array shapes before concatenation. # Added note
    """
   
    if not scaler.obss: 
         raise RuntimeError("Scaler does not appear to be fitted. Call scaler.fit() before create_dataset.")
         
    print(f"Transforming dataset '{lips_dataset.name}' using PowerGridScaler...")
    (inputs_scaled_list, tau_scaled_list), outputs_scaled_list = scaler.transform(lips_dataset)
    print("Transformation complete.")

    print("\n Debug: shapes *before* concatenation ")
    print(f"Number of arrays in inputs_scaled_list: {len(inputs_scaled_list)}")
    for i, arr in enumerate(inputs_scaled_list):
        print(f"  inputs_scaled_list[{i}] shape - {arr.shape}")
    print(f"Number of arrays in tau_scaled_list: {len(tau_scaled_list)}")
    for i, arr in enumerate(tau_scaled_list):
        print(f"  tau_scaled_list[{i}] shape - {arr.shape}")
    print(f"Number of arrays in outputs_scaled_list: {len(outputs_scaled_list)}")
    for i, arr in enumerate(outputs_scaled_list):
        print(f"  outputs_scaled_list[{i}] shape - {arr.shape}")
    print(" End Debug shapes \n")


    try:
        input_arrays_to_concat = inputs_scaled_list + tau_scaled_list
        print(f"Number of input/tau arrays to concat: {len(input_arrays_to_concat)}") 
        inputs_scaled_concat = np.concatenate(input_arrays_to_concat, axis=1)
        
        output_arrays_to_concat = outputs_scaled_list
        print(f"Number of output arrays to concat: {len(output_arrays_to_concat)}") 
        outputs_scaled_concat = np.concatenate(output_arrays_to_concat, axis=1)
        
    except ValueError as e:
        print("Error during concatenation after scaler transform:")
        print(f"Input shapes: {[arr.shape for arr in inputs_scaled_list]}")
        print(f"Tau shapes: {[arr.shape for arr in tau_scaled_list]}")
        print(f"Output shapes: {[arr.shape for arr in outputs_scaled_list]}")
        raise e
        
    print(f"Scaled input shape after concat: {inputs_scaled_concat.shape}") 
    print(f"Scaled output shape after concat: {outputs_scaled_concat.shape}")


    inputs_scaled = tf.constant(inputs_scaled_concat, dtype=tf.float32)
    outputs_scaled = tf.constant(outputs_scaled_concat, dtype=tf.float32)


    num_samples_data = tf.shape(inputs_scaled)[0]
    if not (num_samples_data == tf.shape(YBus_indices)[0] == tf.shape(RBus_edges)[0]):
         raise ValueError("Mismatch in number of samples between data and YBus components")

    dataset = tf.data.Dataset.from_tensor_slices((inputs_scaled, {'edge_indices': YBus_indices, 'RBus_edge': RBus_edges}, outputs_scaled))
    

    if shuffle:
        buffer_size = tf.cast(num_samples_data, dtype=tf.int64) 
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Implements a learning rate schedule with a linear warmup phase
    followed by exponential decay.
    """
    def __init__(self, peak_learning_rate, warmup_steps, decay_steps, decay_rate, name=None):
        """
        Args:
            peak_learning_rate: The target learning rate after warmup.
            warmup_steps: The number of steps for the linear warmup phase.
            decay_steps: The number of steps over which decay is applied once.
            decay_rate: The multiplicative factor for exponential decay.
            name: Optional name for the schedule.
        """
        super().__init__()
        self.peak_learning_rate = tf.cast(peak_learning_rate, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.decay_rate = tf.cast(decay_rate, tf.float32)
        self.name = name

        self.decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.peak_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=False)

    def __call__(self, step):
        """Calculates the learning rate for the current step."""
        step_float = tf.cast(step, tf.float32)

        with tf.name_scope(self.name or "WarmupExponentialDecay"):
            warmup_lr = self.peak_learning_rate*(step_float / self.warmup_steps)
            decay_step = step_float - self.warmup_steps
            decay_lr = self.decay_schedule(decay_step)
            is_warmup = step_float < self.warmup_steps
            learning_rate = tf.cond(is_warmup, lambda: warmup_lr, lambda: decay_lr)
            return learning_rate

    def get_config(self):
        """Required for saving/loading model checkpoints with custom schedule."""
        return {"peak_learning_rate": float(self.peak_learning_rate.numpy()), 
            "warmup_steps": int(self.warmup_steps.numpy()),       
            "decay_steps": int(self.decay_steps.numpy()),      
            "decay_rate": float(self.decay_rate.numpy()),      
            "name": self.name}
    

if __name__ == "__main__":


    CUR_PATH = os.getcwd() 
    USE_CASE = "lips_idf_2023"
    TRAINED_MODELS_DIR = os.path.join(CUR_PATH, "input_data_local", "trained_models", USE_CASE)

    BENCH_CONFIG_PATH = os.path.join(CUR_PATH, "configs", "benchmarks", "lips_idf_2023.ini")
    DATA_PATH = os.path.join(CUR_PATH, "input_data_local", USE_CASE)
    TRAINED_MODELS = os.path.join(CUR_PATH, "input_data_local", "trained_models")
    LOG_PATH = "pinn_transformer_training.log"

    MODEL_NAME = "PinnTransformer_main_file" 
    SAVE_PATH = os.path.join(TRAINED_MODELS_DIR, MODEL_NAME)
    LOAD_WEIGHTS = False
    LOAD_EPOCH_NUMBER = 150


    benchmark_kwargs = {"attr_x": ("prod_p", "prod_v", "load_p", "load_q"),
                        "attr_y": ("a_or", "a_ex", "p_or", "p_ex", "v_or", "v_ex"),
                        "attr_tau": ("line_status", "topo_vect"),
                        "attr_physics": ("YBus",)} # needed for PINN

    
    x_attr_sizes = [62, 62, 99, 99]       
    t_attr_sizes = [186, 540]      
    y_attr_sizes = [186]*6    
    
    print(f"Using x_attr_sizes: {x_attr_sizes} (Sum: {sum(x_attr_sizes)})") 
    print(f"Using t_attr_sizes: {t_attr_sizes} (Sum: {sum(t_attr_sizes)})") 
    print(f"Using y_attr_sizes: {y_attr_sizes} (Sum: {sum(y_attr_sizes)})") 
    
    total_input_dim_observed = sum(x_attr_sizes) + sum(t_attr_sizes)
    print(f"Total input dim based on observed shapes: {total_input_dim_observed}") # should print 1048

    input_feature_dim = sum(x_attr_sizes)

    if total_input_dim_observed != 1048:
         warnings.warn(f"WARNING: Calculated total input dim ({total_input_dim_observed}) "
              f"does not match expected 1048. Check shapes.")

    num_layers = 4 # 2, 4, 8
    d_model = 128
    num_heads = 4
    dff = 512
    learning_rate = 3e-4 # not needed
    epochs = 10
    batch_size = 128 
    save_step = 10 # save every 10 epochs
    peak_learning_rate = 1e-3   
    warmup_epochs = 5           
    decay_rate = 0.96          
    
    print(" Initializing Benchmark ")
    benchmark = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                   config_path=BENCH_CONFIG_PATH,
                                   benchmark_name="Benchmark_competition", 
                                   load_data_set=True,
                                   log_path=LOG_PATH,
                                   load_ybus_as_sparse=True, 
                                   **benchmark_kwargs)

    print(" Initializing Grid2Op Environment ")
    env_kwargs = get_kwargs_simulator_scenario(benchmark.config)
    env = get_env(env_kwargs)
    obs = env.reset() 

    print(" Initializing and Fitting Scaler ")
    scaler = PowerGridScaler()
    train_inputs_raw, train_outputs_raw = benchmark.train_dataset.extract_data(concat=True)
    scaler.fit(benchmark.train_dataset)
    print("Scaler fitted.")
    m_x = scaler._m_x
    sd_x = scaler._sd_x
    m_y = scaler._m_y
    sd_y = scaler._sd_y
    
    unscaler = ProcessMeanStd(m_x, sd_x, m_y, sd_y)
    print("Unscaler initialized.")


    print(" Processing YBus Data ")
    YBus_indices_train, RBus_edges_train = process_ybus(benchmark.train_dataset.data["YBus"])
    YBus_indices_val, RBus_edges_val = process_ybus(benchmark.val_dataset.data["YBus"])
    print("YBus data processed.")

    print(" Creating TensorFlow Datasets ")
    tf_train_dataset = create_dataset(benchmark.train_dataset, scaler, 
                                         YBus_indices_train, RBus_edges_train, 
                                         batch_size, shuffle=True)
    tf_val_dataset = create_dataset(benchmark.val_dataset, scaler,
                                       YBus_indices_val, RBus_edges_val,
                                       batch_size, shuffle=False) 
    print("TensorFlow Datasets created.")


    print(" Building Model ")
    transformer_core = TransformerEncoderTrans(
        output_dim=d_model,
        x_attr_sizes=x_attr_sizes,  
        t_attr_sizes=t_attr_sizes,  
        y_attr_sizes=y_attr_sizes,  
        num_layers=num_layers, 
        d_model=d_model, 
        num_heads=num_heads, 
        dff=dff)
    
    pinn_model = PinnformerModel(
        transformer_core=transformer_core, 
        unscaler=unscaler, 
        obs=obs, 
        x_feature_dim=input_feature_dim, 
        name=MODEL_NAME)

    print("Model built.")
    
    print(" Configuring Checkpoints ")
    
    checkpoint_filepath = os.path.join(SAVE_PATH, "weights_epoch_{epoch:04d}.weights.h5")

    num_train_samples = len(benchmark.train_dataset)
    
    print(f"Number of training samples: {num_train_samples}") 

    steps_per_epoch = num_train_samples//batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1
        warnings.warn(f"Training dataset size ({num_train_samples}) is smaller than batch size ({batch_size}). Checkpoint frequency may behave unexpectedly.")

    save_frequency = save_step*steps_per_epoch 
    print(f"Checkpoint save frequency set to every {save_frequency} batches ({save_step} epochs).")

    warmup_steps = warmup_epochs*steps_per_epoch
    decay_steps = steps_per_epoch
    print(f"LR Schedule: Peak LR={peak_learning_rate}, Warmup Steps={warmup_steps}, Decay Steps={decay_steps}, Decay Rate={decay_rate}")
    
    lr_schedule = WarmupExponentialDecay(peak_learning_rate=peak_learning_rate, warmup_steps=warmup_steps, decay_steps=decay_steps, decay_rate=decay_rate)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,  
        save_freq=save_frequency,
        verbose=1) 

    print("Checkpoint callback configured.")

    print(" Compiling Model ")
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=lr_schedule, # pr pass learning_rate if using schedule
        weight_decay=1e-4)
    
    pinn_model.compile(optimizer=optimizer) 

    # pinn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    # pinn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)) 
    
    print(" Building Model with Sample Batch ") 
    
    sample_batch_dataset = tf_train_dataset.take(1) 
    sample_batch = next(iter(sample_batch_dataset)) 
    
    print(f"Sample batch input shape - {sample_batch[0].shape}") # should be (batch_size, 1048)
    print(f"Sample batch YBus dict keys: {sample_batch[1].keys()}")
    print(f"Sample batch YBus edge_indices shape - {sample_batch[1]['edge_indices'].shape}")
    print(f"Sample batch YBus RBus_edge shape - {sample_batch[1]['RBus_edge'].shape}")
    print(f"Sample batch output shape - {sample_batch[2].shape}") # should be (batch_size, 1116)

    _ = pinn_model(sample_batch[0]) 
    
    pinn_model.summary()
    print("Model compiled and built.")

    initial_epoch_for_fit = 0
    if LOAD_WEIGHTS:
        load_weights_filename = f"weights_epoch_{LOAD_EPOCH_NUMBER:04d}.weights.h5"
        load_weights_path = os.path.join(SAVE_PATH, load_weights_filename)
        
        if os.path.exists(load_weights_path):
            print(f" Loading Weights from: {load_weights_path} ")
            try:
                 pinn_model.load_weights(load_weights_path)
                 initial_epoch_for_fit = LOAD_EPOCH_NUMBER 
                 print(f"Weights loaded successfully. Training will resume from epoch {initial_epoch_for_fit}.")
            except Exception as e:
                 print(f"Error loading weights: {e}. Training from scratch.")
                 initial_epoch_for_fit = 0 # Reset if loading fails
        else:
            print(f"Weight file not found at {load_weights_path}. Training from scratch.")
            initial_epoch_for_fit = 0
    else:
        print("LOAD_WEIGHTS is False. Training from scratch.")
        initial_epoch_for_fit = 0

    TRAIN = True

    if TRAIN:
        print(" Starting Training ")
        history = pinn_model.fit(tf_train_dataset, 
            epochs=epochs, 
            initial_epoch=initial_epoch_for_fit, 
            validation_data=tf_val_dataset,
            callbacks=[model_checkpoint_callback]) 

        
        print(" Training Finished ")

        print(f" Saving Final Model Weights to: {SAVE_PATH} ") 
        os.makedirs(SAVE_PATH, exist_ok=True)

        final_weights_path = os.path.join(SAVE_PATH, "model_weights_final.weights.h5") 
        pinn_model.save_weights(final_weights_path) 
        print(f"Final model weights saved to: {final_weights_path}")

        metadata = {
            "model_name": MODEL_NAME,
            "use_case": USE_CASE,
            "transformer_params": {
                "num_layers": num_layers,
                "d_model": d_model,
                "num_heads": num_heads,
                "dff": dff,
                "x_attr_sizes": x_attr_sizes,
                "t_attr_sizes": t_attr_sizes,
                "y_attr_sizes": y_attr_sizes,
            },
            "training_params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },}

        with open(os.path.join(SAVE_PATH, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, cls=NpEncoder)

        print(f"Model weights and metadata saved in: {SAVE_PATH}")