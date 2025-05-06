import numpy as np
import tensorflow as tf

import os
import pathlib
from typing import Union
import shutil
import json
import tempfile
import importlib

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from lips.augmented_simulators import AugmentedSimulator

from lips.augmented_simulators.tensorflow_simulator import TensorflowSimulator
from lips.logger import CustomLogger
from lips.config import ConfigManager
from lips.dataset import DataSet
from lips.dataset.scaler import Scaler
from lips.utils import NpEncoder

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

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
    heads = tf.split(x, self.num_heads, axis=-1)
    x = tf.stack(heads, axis=1)
    return x

  def call(self, v, k, q, mask=None, training=False):
    seq_len_q = tf.shape(q)[1]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

    # q = tf.keras.layers.GaussianNoise(.1)(q, training=training)
    attention_weights = {}
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (-1, seq_len_q, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation=tf.keras.activations.gelu),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class ReZero(tf.keras.layers.Layer):
    def __init__(self, name):
        super(ReZero, self).__init__(name=name)
        a_init = tf.zeros_initializer()
        self.alpha = tf.Variable(name=self.name + '-alpha',
            initial_value=a_init(shape=(1,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return self.alpha * inputs


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    #self.rz1 = ReZero(self.name+'rz1')
    #self.rz2 = ReZero(self.name+'rz2')

    #self.dropout1 = tf.keras.layers.Dropout(rate)
    #self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training=False, mask=None):

    #inp1 = x
    #x = self.layernorm1(x)
    attn_output, attention_weights = self.mha(x, x, x, mask, training=training)  # (batch_size, input_seq_len, d_model)
    #attn_output = self.dropout1(attn_output, training=training)
    #out1 = x + self.layernorm1(attn_output)  # (batch_size, input_seq_len, d_model)
    # out1 = x + self.rz1(attn_output)
    out1 = self.layernorm1(x + attn_output)

    #out1 = self.layernorm1(out1)
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    #ffn_output = self.dropout2(ffn_output, training=training)
    # out2 = out1 + self.layernorm2(ffn_output)  # (batch_size, input_seq_len, d_model)
    # out2 = out1 + self.rz2(ffn_output)
    out2 = self.layernorm2(out1 + ffn_output)

    return out2, attention_weights


class AttentionPool(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(AttentionPool, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads=2)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.rz = ReZero(self.name+'rz')
        # self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, t, training=False):
        x, _ = self.mha(x, x, t, training=training)
        out1 = tf.math.reduce_mean(x, axis=1)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        #ffn_output = self.dropout2(ffn_output, training=training)
        # out2 = out1 + self.layernorm(ffn_output)  # (batch_size, input_seq_len, d_model)
        out2 = out1 + self.rz(ffn_output)

        return out2


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(CrossAttention, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads=2)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.rz = ReZero(self.name+'rz')
        # self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, q, kv, training=False):
        x, _ = self.mha(kv, kv, q, training=training)

        ffn_output = self.ffn(x)  # (batch_size, q_seq_len, d_model)
        out2 = q + self.rz(ffn_output)

        return out2


class TransformerEncoder(tf.keras.layers.Layer):
  def __init__(self, output_dim, x_attr_sizes, t_attr_sizes, num_layers, d_model, num_heads, dff):
    super(TransformerEncoder, self).__init__()

    self.attr_sizes = list(x_attr_sizes) + list(t_attr_sizes)
    self.n_tau_attr = len(t_attr_sizes)

    self.embeddings = [tf.keras.layers.Dense(d_model) for _ in self.attr_sizes]
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff)
                       for _ in range(num_layers)]

    self.pool = AttentionPool(d_model, dff)
    self.output_layer = point_wise_feed_forward_network(output_dim, dff)

  def call(self, x):
    n = self.n_tau_attr

    x = tf.split(x, self.attr_sizes, axis=1)
    x = [tf.expand_dims(lay(inp), axis=1) for (lay, inp) in zip(self.embeddings, x)]
    x, t = x[:-n], x[-n:]
    x, t = tf.concat(x, axis=1), tf.concat(t, axis=1)
    x = tf.concat([t, x], axis=1)

    x = self.ffn(x)

    for lay in self.enc_layers:
      x, _ = lay(x)

    x = self.pool(x[:, :n], x[:, :n])
    x = self.output_layer(x)
    return x


from leap_net.LtauNoAdd import LtauNoAdd


class TransformerEncoder(tf.keras.layers.Layer):
  def __init__(self, output_dim, x_attr_sizes, t_attr_sizes, y_attr_sizes,
               num_layers, d_model, num_heads, dff):
    super(TransformerEncoder, self).__init__()

    self.attr_sizes = list(x_attr_sizes) + list(t_attr_sizes)

    self.embeddings = [tf.keras.layers.Dense(d_model) for _ in self.attr_sizes]
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff)
                       for _ in range(num_layers)]
    self.leap_layer = LtauNoAdd()

    self.output_emb_layer = point_wise_feed_forward_network(d_model, dff)
    self.output_layer = point_wise_feed_forward_network(output_dim, dff)

  def call(self, x):
    x = tf.split(x, self.attr_sizes, axis=1)
    x = [tf.expand_dims(lay(inp), axis=1)
         for (lay, inp) in zip(self.embeddings, x)]
    x = tf.concat(x, axis=1)

    x = self.ffn(x)

    for lay in self.enc_layers:
      x, _ = lay(x)

    x = tf.reduce_max(x, axis=1)
    x = self.output_emb_layer(x)
    x = self.leap_layer([x, x])

    x = self.output_layer(x)
    return x


class TransformerEncoderTrans(tf.keras.layers.Layer):
  def __init__(self, output_dim, x_attr_sizes, t_attr_sizes, y_attr_sizes,
               num_layers, d_model, num_heads, dff):
    super(TransformerEncoderTrans, self).__init__()

    self.attr_sizes = list(x_attr_sizes) + list(t_attr_sizes)

    self.seq_len = len(y_attr_sizes)
    print('attr_sizes', x_attr_sizes, t_attr_sizes, y_attr_sizes)
    self.latent_emb = tf.keras.layers.Embedding(len(y_attr_sizes), d_model)
    self.cross = CrossAttention(d_model, dff)

    self.embeddings = [tf.keras.layers.Dense(d_model, activation='elu')
                       for _ in self.attr_sizes]

    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.d_model = d_model

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff)
                       for _ in range(num_layers)]

    self.dec_emb_layer = point_wise_feed_forward_network(output_dim, dff)
    self.dec_layers = [point_wise_feed_forward_network(s, 2*s)
                       for s in y_attr_sizes]

  def call(self, inp):
    split = tf.split(inp, self.attr_sizes, axis=1)
    x = [tf.expand_dims(lay(inp), axis=1)
         for (lay, inp) in zip(self.embeddings, split)]
    x = tf.concat(x, axis=1)
    x = self.ffn(x)
    print("inp", x.shape)

    lat = self.latent_emb(tf.range(self.seq_len)[None])
    x = self.cross(lat, x)

    print('cross', x.shape)
    for lay in self.enc_layers:
      x, _ = lay(x)

    x = self.dec_emb_layer(x)
    split = tf.unstack(x, axis=1)
    x = [lay(z) for (lay, z) in zip(self.dec_layers, split)]
    x = tf.concat(x, axis=-1)
    return x

# %%writefile /content/LIPS/lips/augmented_simulators/tensorflow_models/transformer.py

# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import pathlib
from typing import Union
import json
import warnings

import numpy as np
# from leap_net import ResNetLayer

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow import keras



# from code.transformer import TransformerEncoder, TransformerEncoderTrans

import os
import pathlib
from typing import Union
import json
import warnings

import numpy as np
# from leap_net import ResNetLayer

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow import keras

class SimpNet(TensorflowSimulator):
    """Fully Connected architecture
    Parameters
    ----------
    sim_config_path : ``str``
        The path to the configuration file for simulator.
        It should contain all the required hyperparameters for this model.
    sim_config_name : Union[str, None], optional
        _description_, by default None
    name : Union[str, None], optional
        _description_, by default None
    scaler : Union[Scaler, None], optional
        _description_, by default None
    bench_config_path : Union[str, pathlib.Path, None], optional
        _description_, by default None
    bench_config_name : Union[str, None], optional
        _description_, by default None
    log_path : Union[None, str], optional
        _description_, by default None
    Raises
    ------
    RuntimeError
        _description_
    """
    def __init__(self,
                 sim_config_path: str,
                 bench_config_path: Union[str, pathlib.Path],
                 bench_config_name: Union[str, None]=None,
                 bench_kwargs: dict={},
                 sim_config_name: Union[str, None]=None,
                 name: Union[str, None]=None,
                 scaler: Union[Scaler, None]=None,
                 log_path: Union[None, str]=None,
                 **kwargs):
        super().__init__(name=name, log_path=log_path, **kwargs)
        if not os.path.exists(sim_config_path):
            raise RuntimeError("Configuration path for the simulator not found!")
        if not str(sim_config_path).endswith(".ini"):
            raise RuntimeError("The configuration file should have `.ini` extension!")
        sim_config_name = sim_config_name if sim_config_name is not None else "DEFAULT"
        self.sim_config = ConfigManager(section_name=sim_config_name, path=sim_config_path)
        self.bench_config = ConfigManager(section_name=bench_config_name, path=bench_config_path)
        self.bench_config.set_options_from_dict(**bench_kwargs)
        self.name = name if name is not None else self.sim_config.get_option("name")
        self.name = self.name + '_' + sim_config_name
        # scaler
        self.scaler = scaler() if scaler else None
        # Logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # model parameters
        self.params = self.sim_config.get_options_dict()
        self.params.update(kwargs)
        # Define layer to be used for the model
        self.layers = {"linear": keras.layers.Dense}
        self.layer = self.layers.get(self.params["layer"], None)
        if self.layer is None:
            self.layer = keras.layers.Dense

        # optimizer
        if "lr" in kwargs:
            if not isinstance(kwargs["lr"], float):
                raise RuntimeError("Learning rate (lr) is provided, it should be a float")
            lr = kwargs['lr']
        else:
            lr = self.params["optimizer"]["params"]["lr"]
        print('lr', lr)
        self._optimizer = keras.optimizers.Adam(learning_rate=lr)

        self._model: Union[keras.Model, None] = None

        self.input_size = None if kwargs.get("input_size") is None else kwargs["input_size"]
        self.output_size = None if kwargs.get("output_size") is None else kwargs["output_size"]

    def build_model(self):
        """Build the model
        Returns
        -------
        Model
            _description_
        """
        super().build_model()
        input_ = keras.layers.Input(shape=(self.input_size,), name="input")
        x = input_
        # x = keras.layers.Dropout(rate=self.params["input_dropout"], name="input_dropout")(x)
        for layer_id, layer_size in enumerate(self.params["layers"]):
            x = self.layer(layer_size, name=f"layer_{layer_id}")(x)
            x = keras.layers.LayerNormalization(name=f"norm_{layer_id}")(x)
            x = keras.layers.Activation(self.params["activation"], name=f"activation_{layer_id}")(x)
            # x = keras.layers.Activation(self.params["activation"], name=f"activation_{layer_id}")(x)
            # x = keras.layers.Dropout(rate=self.params["dropout"], name=f"dropout_{layer_id}")(x)

        output_ = keras.layers.Dense(self.output_size)(x)
        self._model = keras.Model(inputs=input_,
                                  outputs=output_,
                                  name=f"{self.name}_model")
        return self._model

    def process_dataset(self, dataset: DataSet, training: bool=False) -> tuple:
        """process the datasets for training and evaluation
        This function transforms all the dataset into something that can be used by the neural network (for example)
        Warning
        -------
        It works with StandardScaler only for the moment.
        Parameters
        ----------
        dataset : DataSet
            _description_
        Scaler : bool, optional
            _description_, by default True
        training : bool, optional
            _description_, by default False
        Returns
        -------
        tuple
            the normalized dataset with features and labels
        """
        if training:
            self._infer_size(dataset)
            inputs, outputs = dataset.extract_data(concat=False)

            inputs, outputs = dataset.extract_data(concat=True)
            if self.scaler is not None:
                inputs, outputs = self.scaler.fit_transform(inputs, outputs)
        else:
            inputs, outputs = dataset.extract_data(concat=True)
            if self.scaler is not None:
                inputs, outputs = self.scaler.transform(inputs, outputs)

        return inputs, outputs

    def _infer_size(self, dataset: DataSet):
        """Infer the size of the model
        Parameters
        ----------
        dataset : DataSet
            _description_
        Returns
        -------
        None
            _description_
        """
        *dim_inputs, self.output_size = dataset.get_sizes()
        self.input_size = np.sum(dim_inputs)

    def _post_process(self, dataset, predictions):
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        predictions = super()._post_process(dataset, predictions)
        return predictions

    def _save_metadata(self, path: str):
        super()._save_metadata(path)
        if self.scaler is not None:
            self.scaler.save(path)
        res_json = {}
        res_json["input_size"] = self.input_size
        res_json["output_size"] = self.output_size
        with open((path / "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def _load_metadata(self, path: str):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        super()._load_metadata(path)
        if self.scaler is not None:
            self.scaler.load(path)
        with open((path / "metadata.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self.input_size = res_json["input_size"]
        self.output_size = res_json["output_size"]


class TensorflowSimulator(AugmentedSimulator):
    """_summary_

        Parameters
        ----------
        name : str, optional
            _description_, by default None
        config : ConfigManager
            _description_
        """
    def __init__(self,
                 name: Union[str, None]=None,
                 log_path: Union[str, None] = None,
                 **kwargs):
        super().__init__(name=name, log_path=log_path, **kwargs)
        # logger
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger
        self._optimizer = None

        self.input_size = None
        self.output_size = None

        # setting seeds
        np.random.seed(1)
        # tf.random.set_seed(2)


    def build_model(self):
        """build tensorflow model

        Parameters
        ----------
        **kwargs : dict
            if parameters indicated, it will replace config parameters

        Returns
        -------
        keras.Model
            _description_
        """
        if self.input_size is None or self.output_size is None:
            raise RuntimeError("input_size is not set")


    def train(self,
              train_dataset: DataSet,
              val_dataset: Union[None, DataSet] = None,
              save_path: Union[None, str] = None,
              **kwargs):
        """Function used to train a neural network

        Parameters
        ----------
        train_dataset : DataSet
            training dataset
        val_dataset : Union[None, DataSet], optional
            validation dataset, by default None
        save_path : Union[None, str], optional
            the path where the trained model should be saved, by default None
            #TODO: a callback for tensorboard and another for saving the model
        """
        super().train(train_dataset, val_dataset)
        self.params.update(kwargs)
        processed_x, processed_y = self.process_dataset(train_dataset, training=True)

        if val_dataset is not None:
            processed_x_val, processed_y_val = self.process_dataset(val_dataset, training=False)
            validation_data = (processed_x_val, processed_y_val)
        else:
            validation_data = None

        # init the model
        self.build_model()

        self._model.compile(optimizer=self._optimizer,
                            loss="mae", # self.params["loss"]["name"],
                            metrics=self.params["metrics"])

        cb = [
            tf.keras.callbacks.ModelCheckpoint(
                "temp.keras",
                monitor="val_mae",
                verbose=1,
                save_best_only=True,
                save_weights_only=True)
            ]

        self.logger.info("Training of {%s} started", self.name)
        history_callback = self._model.fit(x=processed_x,
                                           y=processed_y,
                                           validation_data=validation_data,
                                           epochs=self.params["epochs"],
                                           batch_size=self.params["train_batch_size"],
                                           shuffle=self.params["shuffle"],
                                           callbacks=cb,
                                           verbose=2)
        self.logger.info("Training of {%s} finished", self.name)
        self.write_history(history=history_callback, val_dataset=validation_data)
        self.trained = True
        self._model.load_weights("temp.keras")
        os.remove("temp.keras")

        if save_path is not None:
            self.save(save_path)

        return history_callback

    def predict(self, dataset: DataSet, **kwargs) -> dict:
        """_summary_

        Parameters
        ----------
        dataset : DataSet
            test datasets to evaluate
        """
        super().predict(dataset)

        if "eval_batch_size" in kwargs:
            self.params["eval_batch_size"] = kwargs["eval_batch_size"]
        # self.params.update(kwargs)

        #processed_x, processed_y = self._process_all_dataset(dataset, training=False)
        processed_x, _ = self.process_dataset(dataset, training=False)

        # make the predictions
        predictions = self._model.predict(processed_x, batch_size=self.params["eval_batch_size"])

        predictions = self._post_process(dataset, predictions)

        self._predictions[dataset.name] = predictions
        self._observations[dataset.name] = dataset.data

        return predictions

    def process_dataset(self, dataset: DataSet, training: bool) -> tuple:
        """process the datasets for training and evaluation

        each augmented simulator requires its owan data preparation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Parameters
        ----------
        dataset : DataSet
            _description_
        training : bool, optional
            _description_, by default False

        Returns
        -------
        tuple
            the normalized dataset with features and labels
        """
        super().process_dataset(dataset, training)
        inputs, outputs = dataset.extract_data()

        return inputs, outputs

    def _post_process(self, dataset, predictions):
        """Do some post processing on the predictions

        Parameters
        ----------
        predictions : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return dataset.reconstruct_output(predictions)


    ###############################################
    # function used to save and restore the model #
    ###############################################
    def save(self, path: str, save_metadata: bool=True):
        """_summary_

        Parameters
        ----------
        path : str
            _description_
        save_metadata : bool, optional
            _description_, by default True
        """
        save_path =  pathlib.Path(path) / self.name
        super().save(save_path)

        self._save_model(save_path)

        if save_metadata:
            self._save_metadata(save_path)

        self.logger.info("Model {%s} is saved at {%s}", self.name, save_path)

    def _save_model(self, path: Union[str, pathlib.Path], ext: str=".h5"):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        file_name = path / ("weights" + ext)
        self._model.save_weights(file_name)

    def _save_metadata(self, path: Union[str, pathlib.Path]):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        # for json serialization of paths
        #pydantic.json.ENCODERS_BY_TYPE[pathlib.PosixPath] = str
        #pydantic.json.ENCODERS_BY_TYPE[pathlib.WindowsPath] = str
        self._save_losses(path)
        with open((path / "config.json"), "w", encoding="utf-8") as f:
            json.dump(obj=self.params, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def restore(self, path: str):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        full_path = path / self.name
        if not full_path.exists():
            raise FileNotFoundError(f"path {full_path} not found")
        # load the metadata
        self._load_metadata(full_path)
        self._load_model(full_path)

        self.logger.info("Model {%s} is loaded from {%s}", self.name, full_path)

    def _load_model(self, path: str):
        nm_file = "weights.h5"
        path_weights = path / nm_file
        if not path_weights.exists():
            raise FileNotFoundError(f"Weights file {path_weights} not found")
        self.build_model()
        # load the weights
        with tempfile.TemporaryDirectory() as path_tmp:
            nm_tmp = os.path.join(path_tmp, nm_file)
            # copy the weights into this file
            shutil.copy(path_weights, nm_tmp)
            # load this copy (make sure the proper file is not corrupted even if the loading fails)
            self._model.load_weights(nm_tmp)

    def _load_metadata(self, path: str):
        """
        load the model metadata
        """
        # load scaler parameters
        #self.scaler.load(full_path)
        self._load_losses(path)
        with open((path / "config.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self.params.update(res_json)
        return self.params

    def _save_losses(self, path: Union[str, pathlib.Path]):
        """
        save the losses
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        res_losses = {}
        res_losses["train_losses"] = self.train_losses
        res_losses["train_metrics"] = self.train_metrics
        res_losses["val_losses"] = self.val_losses
        res_losses["val_metrics"] = self.val_metrics
        with open((path / "losses.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_losses, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def _load_losses(self, path: Union[str, pathlib.Path]):
        """
        load the losses
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        with open((path / "losses.json"), "r", encoding="utf-8") as f:
            res_losses = json.load(fp=f)
        self.train_losses = res_losses["train_losses"]
        self.train_metrics = res_losses["train_metrics"]
        self.val_losses = res_losses["val_losses"]
        self.val_metrics = res_losses["val_metrics"]

    #########################
    # Some Helper functions #
    #########################
    def summary(self):
        """summary of the model
        """
        print(self._model.summary())

    def plot_model(self, path: Union[str, None]=None, file_name: str="model"):
        """Plot the model architecture using GraphViz Library

        """
        # verify if GraphViz and pydot are installed
        pydot_found = importlib.util.find_spec("pydot")
        graphviz_found = importlib.util.find_spec("graphviz")
        if pydot_found is None or graphviz_found is None:
            raise RuntimeError("pydot and graphviz are required to use this function")

        if not pathlib.Path(path).exists():
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        tf.keras.utils.plot_model(
            self._model,
            to_file=file_name+".png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=56,
            layer_range=None,
            show_layer_activations=False,
        )

    def write_history(self, history: dict, val_dataset=None):
        """write the history of the training

        Parameters
        ----------
        history_callback : keras.callbacks.History
            the history of the training
        """
        self.train_losses = history.history["loss"]
        if val_dataset is not None:
            self.val_losses = history.history["val_loss"]

        for metric in self.params["metrics"]:
            self.train_metrics[metric] = history.history[metric]
            if val_dataset is not None:
                self.val_metrics[metric] = history.history["val_" + metric]

    def count_parameters(self):
        """count the number of parameters of the model

        Returns
        -------
        int
            the number of parameters
        """
        return self._model.count_params()

    def visualize_convergence(self, figsize=(15,5), save_path: str=None):
        """Visualizing the convergence of the model
        """
        # raise an error if the train_losses is empty
        if len(self.train_losses) == 0:
            raise RuntimeError("The model should be trained before visualizing the convergence")
        num_metrics = len(self.params["metrics"])
        if num_metrics == 0:
            nb_subplots = 1
        else:
            nb_subplots = num_metrics + 1
        fig, ax = plt.subplots(1,nb_subplots, figsize=figsize)
        ax[0].set_title("MSE")
        ax[0].plot(self.train_losses, label='train_loss')
        if len(self.val_losses) > 0:
            ax[0].plot(self.val_losses, label='val_loss')
        for idx_, metric_name in enumerate(self.params["metrics"]):
            ax[idx_+1].set_title(metric_name)
            ax[idx_+1].plot(self.train_metrics[metric_name], label=f"train_{metric_name}")
            if len(self.val_metrics[metric_name]) > 0:
                ax[idx_+1].plot(self.val_metrics[metric_name], label=f"val_{metric_name}")
        for i in range(nb_subplots):
            ax[i].grid()
            ax[i].legend()
        # save the figure
        if save_path is not None:
            if not pathlib.Path(save_path).exists():
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
