# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural Replicator Dynamics [Omidshafiei et al, 2019].

A policy gradient-like extension to replicator dynamics and the hedge algorithm
that incorporates function approximation.

# References

Shayegan Omidshafiei, Daniel Hennes, Dustin Morrill, Remi Munos,
  Julien Perolat, Marc Lanctot, Audrunas Gruslys, Jean-Baptiste Lespiau,
  Karl Tuyls. Neural Replicator Dynamics. https://arxiv.org/abs/1906.00190.
  2019.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import rcfr


def thresholded(logits, regrets, threshold=2.0):
  """Zeros out `regrets` where `logits` are too negative or too large."""
  can_decrease = tf.cast(tf.greater(logits, -threshold), tf.float32)
  can_increase = tf.cast(tf.less(logits, threshold), tf.float32)
  regrets_negative = tf.minimum(regrets, 0.0)
  regrets_positive = tf.maximum(regrets, 0.0)
  return can_decrease * regrets_negative + can_increase * regrets_positive


# @tf.function
def train(model,
          data,
          batch_size,
          step_size=1.0,
          threshold=2.0,
          alpha=1.0,
          random_shuffle_size=None,
          autoencoder_loss=None):
  """Train NeuRD `model` on `data`."""
  if random_shuffle_size is None:
    random_shuffle_size = 10 * batch_size
  data = data.shuffle(random_shuffle_size)
  data = data.batch(batch_size)
  data = data.repeat(1)

  for x, regrets in data:
    with tf.GradientTape() as tape:
      output = model(x, training=True)
      logits = output[:, :1]
      logits = logits - tf.reduce_mean(logits, keepdims=True)

      regrets = tf.stop_gradient(
          thresholded(logits, regrets, threshold=threshold))
      utility = tf.reduce_mean(logits * regrets)

      if autoencoder_loss is not None:
        utility = utility - autoencoder_loss(x, output[:, 1:])

    grad = tape.gradient(utility, model.trainable_variables)
    # print(regrets)

    for i, var in enumerate(model.trainable_variables):
      var.assign_add(step_size * grad[i])


class DeepNeurdModel(object):
  """A flexible deep feedforward NeuRD model class.

  Properties:
    layers: The `tf.keras.Layer` layers describing this  model.
    trainable_variables: The trainable `tf.Variable`s in this model's `layers`.
    losses: This model's layer specific losses (e.g. regularizers).
  """

  def __init__(self,
               game,
               num_hidden_units,
               num_hidden_layers=1,
               num_hidden_factors=0,
               hidden_activation=tf.nn.relu,
               use_skip_connections=False,
               regularizer=None,
               autoencode=False):
    """Creates a new `DeepNeurdModel.

    Args:
      game: The OpenSpiel game being solved.
      num_hidden_units: The number of units in each hidden layer.
      num_hidden_layers: The number of hidden layers. Defaults to 1.
      num_hidden_factors: The number of hidden factors or the matrix rank of the
        layer. If greater than zero, hidden layers will be split into two
        separate linear transformations, the first with
        `num_hidden_factors`-columns and the second with
        `num_hidden_units`-columns. The result is that the logical hidden layer
        is a rank-`num_hidden_units` matrix instead of a rank-`num_hidden_units`
        matrix. When `num_hidden_units < num_hidden_units`, this is effectively
        implements weight sharing. Defaults to 0.
      hidden_activation: The activation function to apply over hidden layers.
        Defaults to `tf.nn.relu`.
      use_skip_connections: Whether or not to apply skip connections (layer
        output = layer(x) + x) on hidden layers. Zero padding or truncation is
        used to match the number of columns on layer inputs and outputs.
      regularizer: A regularizer to apply to each layer. Defaults to `None`.
      autoencode: Whether or not to output a reconstruction of the inputs upon
        being called. Defaults to `False`.
    """

    self._autoencode = autoencode
    self._use_skip_connections = use_skip_connections
    self._hidden_are_factored = num_hidden_factors > 0

    self.layers = []
    for _ in range(num_hidden_layers):
      if self._hidden_are_factored:
        self.layers.append(
            tf.keras.layers.Dense(
                num_hidden_factors,
                use_bias=True,
                kernel_regularizer=regularizer))

      self.layers.append(
          tf.keras.layers.Dense(
              num_hidden_units,
              use_bias=True,
              activation=hidden_activation,
              kernel_regularizer=regularizer))

    self.layers.append(
        tf.keras.layers.Dense(
            1 + self._autoencode * rcfr.num_features(game),
            use_bias=True,
            kernel_regularizer=regularizer))

    # Construct variables for all layers by exercising the network.
    x = tf.zeros([1, rcfr.num_features(game)])
    for layer in self.layers:
      x = layer(x)

    self.trainable_variables = sum(
        [layer.trainable_variables for layer in self.layers], [])
    self.losses = sum([layer.losses for layer in self.layers], [])

  def __call__(self, x, training=False):
    """Evaluates this model on x.

    Args:
      x: Model input.
      training: Whether or not this is being called during training. If
        `training` and the constructor argument `autoencode` was `True`, then
        the output will contain the estimated regrets concatenated with a
        reconstruction of the input, otherwise only regrets will be returned.
        Defaults to `False`.

    Returns:
      The `tf.Tensor` resulting from evaluating this model on `x`. If
        `training` and the constructor argument `autoencode` was `True`, then
        it will contain the estimated regrets concatenated with a
        reconstruction of the input, otherwise only regrets will be returned.
    """
    y = rcfr.feedforward_evaluate(
        layers=self.layers,
        x=x,
        use_skip_connections=self._use_skip_connections,
        hidden_are_factored=self._hidden_are_factored)
    return y if training else y[:, :1]


class CounterfactualNeurdSolver(object):
  """All-actions, strong NeuRD on counterfactual regrets.

  No regularization bonus is applied, so the current policy likely will not
  converge. The average policy profile is updated and stored in a full
  game-size table and may converge to an approximate Nash equilibrium in
  two-player, zero-sum games.
  """

  def __init__(self, game, alpha, models, session=None):
    """Creates a new `CounterfactualNeurdSolver`.

    Args:
      game: An OpenSpiel `Game`.
      models: Current policy models (optimizable array-like -> `tf.Tensor`
        callables) for both players.
      session: A TensorFlow `Session` to convert sequence weights from
        `tf.Tensor`s produced by `models` to `np.array`s. If `None`, it is
        assumed that eager mode is enabled. Defaults to `None`.
    """
    self._game = game
    self._alpha = alpha
    self._models = models
    self._root_wrapper = rcfr.RootStateWrapper(game.new_initial_state())
    self._session = session

    self._cumulative_seq_probs = [
        np.zeros(n) for n in self._root_wrapper.num_player_sequences
    ]

  def _sequence_weights(self, alpha=None, increase=None, gamma=None, adaptive_policy=None, total_iteration=None, player=None, current_iteration=None, exploit_rate=None, semi_percent=None, conv=None, exp_exploit_rate=None):
    """Returns exponentiated weights for each sequence as an `np.array`."""
    if player is None:
      return [
          self._sequence_weights(alpha=alpha, increase=increase, gamma=gamma, adaptive_policy=adaptive_policy, total_iteration=total_iteration,  player=player, current_iteration=current_iteration, exploit_rate=exploit_rate, semi_percent=semi_percent, conv=conv, exp_exploit_rate=exp_exploit_rate)
          for player in range(self._game.num_players())
      ]
    else:
      tensor = tf.squeeze(self._models[player](
          self._root_wrapper.sequence_features[player]))

      # tensor = tensor - tf.reduce_max(tensor, keepdims=True)
      # tensor = tf.math.exp(tensor)
      # print(tf.reduce_sum(tensor, keepdims=True))
      # tensor = tf.nn.softmax(tensor)
      # stacked_tensor = tf.stack([tensor, tensor])
      # stacked_tensor = tf.transpose(stacked_tensor)
      # print(tensor)
      # tensor = tf.nn.softmax(tensor)

      length = tf.shape(tensor)[0]
      stacked_tensor = tf.reshape(tensor,[1,length])
    #   tsallis = TsallisLoss(alpha=self._alpha)

    

      if current_iteration:
        if alpha == 1:
          adaptive_alpha = alpha
        elif adaptive_policy == 1:
          adaptive_alpha = self._linear_update(large_alpha=alpha, current_iteration=current_iteration, total_iteration=total_iteration, increase=increase, semi_percent=semi_percent)
        elif adaptive_policy == 2:
          adaptive_alpha = self._exp_update(large_alpha=alpha, current_iteration=current_iteration, total_iteration=total_iteration, increase=increase, gamma=gamma)
        elif adaptive_policy == 3:
          adaptive_alpha = self._exploit_update(conv=conv, exploit_rate=exploit_rate)
        elif adaptive_policy == 4:
          adaptive_alpha = self._exp_exploit_update(conv=conv, exp_exploit_rate=exp_exploit_rate)
        else:
          print("ERROR: Policy should be either linear or exp")

      else:
        adaptive_alpha = alpha

      # print(adaptive_alpha)
      # print(conv)

      tsallis = TsallisLoss(alpha=adaptive_alpha)
      tensor = tsallis.predict(stacked_tensor.numpy())[0] 
      num_zeros = np.nonzero(tensor)[0].shape[0]

      # print(float(num_zeros)/tensor.shape[0])

      return tensor if self._session is None else self._session(tensor)

  def _linear_update(self, large_alpha, current_iteration, total_iteration, increase, semi_percent):
    if semi_percent == 0.5:
      if increase:
        alpha = (float(current_iteration) / total_iteration) * (large_alpha-1) + 1.0
      else:
        alpha = large_alpha - (float(current_iteration) / total_iteration) * (large_alpha-1)
    else:
      if increase:
        if current_iteration <= 0.5*total_iteration:
          alpha = (float(current_iteration) / total_iteration) * 2 * semi_percent * (large_alpha-1) + 1.0
        else:
          alpha = 1.0 + (large_alpha-1)*semi_percent + (float(current_iteration-0.5*total_iteration)*2 / total_iteration) * (large_alpha-1-(large_alpha-1)*semi_percent )
      else:
        print("DO not use decrease with semi!")
    
    return min(max(1, round(alpha, 3)), 1.5)
  

  def _exploit_update(self, conv, exploit_rate):
    alpha = 1 + (1.0/(exploit_rate*conv))

    return min(max(1, round(alpha, 3)), 1.5)

  
  def _exp_exploit_update(self, conv, exp_exploit_rate):
    alpha = 1 + (1.0/(exp_exploit_rate*math.exp(-conv)))

    return min(max(1, round(alpha, 3)), 1.5)


  def _exp_update(self, large_alpha, current_iteration, total_iteration, gamma, increase):
    # print("large_alpha: ", large_alpha)
    # print("current_iteration: ", current_iteration)
    # print("total_iteration: ", total_iteration)
    # print("gamma", gamma)

    num_change = math.log(1.0/(large_alpha))/math.log(gamma)
    frequent = max(int(total_iteration/num_change), 1)
    times = int(current_iteration/frequent)
    # print("times: ", times)

    if increase:
      alpha = 1 * ((1.0/gamma)**times)
    else:
      alpha = large_alpha * (gamma**times)

    
    return min(max(1, round(alpha, 3)), 1.5)




  def current_policy(self):
    """Returns the current policy profile.

    Returns:
      A `dict<info state, list<Action, probability>>` that maps info state
      strings to `Action`-probability pairs describing each player's policy.
    """
    return self._root_wrapper.sequence_weights_to_tabular_profile(
        self._sequence_weights())

  def average_policy(self):
    """Returns the average of all policies iterated.

    The policy is computed using the accumulated policy probabilities computed
    using `evaluate_and_update_policy`.

    Returns:
      A `dict<info state, list<Action, probability>>` that maps info state
      strings to (Action, probability) pairs describing each player's policy.
    """
    return self._root_wrapper.sequence_weights_to_tabular_profile(
        self._cumulative_seq_probs)

  def _previous_player(self, player):
    """The previous player in the turn ordering."""
    return player - 1 if player > 0 else self._game.num_players() - 1

  def _average_policy_update_player(self, regret_player):
    """The player for whom the average policy should be updated."""
    return self._previous_player(regret_player)

  def evaluate_and_update_policy(self, train_fn, alpha, increase=None, gamma=None, adaptive_policy=None, total_iteration=None, current_iteration=None, semi_percent=None, exploit_rate=None, conv=None, exp_exploit_rate=None):
    """Performs a single step of policy evaluation and policy improvement.

    Args:
      train_fn: A (model, `tf.data.Dataset`) function that trains the given
        regression model to accurately reproduce the x to y mapping given x-y
        data.
    """
    sequence_weights = self._sequence_weights(current_iteration=current_iteration, alpha=alpha, increase=increase, gamma=gamma, adaptive_policy=adaptive_policy, total_iteration=total_iteration, semi_percent=semi_percent, exploit_rate=exploit_rate, conv=conv, exp_exploit_rate=exp_exploit_rate)
    player_seq_features = self._root_wrapper.sequence_features
    for regret_player in range(self._game.num_players()):
      seq_prob_player = self._average_policy_update_player(regret_player)

      regrets, seq_probs = (
          self._root_wrapper.counterfactual_regrets_and_reach_weights(
              regret_player, seq_prob_player, alpha, *sequence_weights))

      self._cumulative_seq_probs[seq_prob_player] += seq_probs

      targets = tf.expand_dims(regrets.astype('float32'), axis=1)
      data = tf.data.Dataset.from_tensor_slices(
          (player_seq_features[regret_player], targets))

      regret_player_model = self._models[regret_player]
      train_fn(regret_player_model, data)
      sequence_weights[regret_player] = self._sequence_weights(player=regret_player, current_iteration=current_iteration, alpha=alpha, increase=increase, gamma=gamma, adaptive_policy=adaptive_policy, total_iteration=total_iteration, semi_percent=semi_percent, exploit_rate=exploit_rate, conv=conv, exp_exploit_rate=exp_exploit_rate)


# Author: Mathieu Blondel
# License: Simplified BSD

"""
NumPy implementation of

Learning Classifiers with Fenchel-Young Losses:
    Generalized Entropies, Margins, and Algorithms.
Mathieu Blondel, André F. T. Martins, Vlad Niculae.
https://arxiv.org/abs/1805.09717
"""


import numpy as np


def conjugate_function(theta, grad, Omega):
    return np.sum(theta * grad, axis=1) - Omega(grad)


class FYLoss(object):

    def __init__(self, weights="average"):
        self.weights = weights

    def forward(self, y_true, theta):
        y_true = np.array(y_true)

        self.y_pred = self.predict(theta)
        ret = conjugate_function(theta, self.y_pred, self.Omega)

        if len(y_true.shape) == 2:
            # y_true contains label proportions
            ret += self.Omega(y_true)
            ret -= np.sum(y_true * theta, axis=1)

        elif len(y_true.shape) == 1:
            # y_true contains label integers (0, ..., n_classes-1)

            if y_true.dtype != np.long:
                raise ValueError("y_true should contains long integers.")

            all_rows = np.arange(y_true.shape[0])
            ret -= theta[all_rows, y_true]

        else:
            raise ValueError("Invalid shape for y_true.")

        if self.weights == "average":
            return np.mean(ret)
        else:
            return np.sum(ret)


    def __call__(self, y_true, theta):
        return self.forward(y_true, theta)


class SquaredLoss(FYLoss):

    def Omega(self, mu):
        return 0.5 * np.sum((mu ** 2), axis=1)

    def predict(self, theta):
        return theta


class PerceptronLoss(FYLoss):

    def predict(self, theta):
        theta = np.array(theta)
        ret = np.zeros_like(theta)
        all_rows = np.arange(theta.shape[0])
        ret[all_rows, np.argmax(theta, axis=1)] = 1
        return ret

    def Omega(self, theta):
        return np.zeros(len(theta))


def Shannon_negentropy(p, axis):
    p = np.array(p)
    tmp = np.zeros_like(p)
    mask = p > 0
    tmp[mask] = p[mask] * np.log(p[mask])
    return np.sum(tmp, axis)


class LogisticLoss(FYLoss):

    def predict(self, theta):
        exp_theta = np.exp(theta - np.max(theta, axis=1)[:, np.newaxis])
        return exp_theta / np.sum(exp_theta, axis=1)[:, np.newaxis]

    def Omega(self, p):
        return Shannon_negentropy(p, axis=1)


class Logistic_OVA_Loss(FYLoss):

    def predict(self, theta):
        return 1. / (1 + np.exp(-theta))

    def Omega(self, p):
        return Shannon_negentropy(p, axis=1) + Shannon_negentropy(1 - p, axis=1)


def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2

    z: float or array
        If array, len(z) must be compatible with V

    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    V = np.array(V)

    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


class SparsemaxLoss(FYLoss):

    def predict(self, theta):
        return projection_simplex(theta, axis=1)

    def Omega(self, p):
        p = np.array(p)
        return 0.5 * np.sum((p ** 2), axis=1) - 0.5


# FIXME: implement bisection in Numba.
def _bisection(theta, omega_p, omega_p_inv, max_iter=20, tol=1e-3):
    theta = np.array(theta)
    t_min = np.max(theta, axis=1) - omega_p(1.0)
    t_max = np.max(theta, axis=1) - omega_p(1.0 / theta.shape[1])
    p = np.zeros_like(theta)

    for i in range(len(theta)):

        thresh = omega_p(0)

        for it in range(max_iter):
            t = (t_min[i] + t_max[i]) / 2.0
            p[i] = omega_p_inv(np.maximum(theta[i] - t, thresh))
            f = np.sum(p[i]) - 1
            if f < 0:
                t_max[i] = t
            else:
                t_min[i] = t
            if np.abs(f) < tol:
                break

    return p


class TsallisLoss(FYLoss):

    def __init__(self, alpha=1.5, max_iter=20, tol=1e-3, weights="average"):
        if alpha < 1:
            raise ValueError("alpha should be greater or equal to 1.")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights = weights

    def predict(self, theta):
        # Faster algorithms for specific cases.
        if self.alpha == 1:
            return LogisticLoss().predict(theta)

        if self.alpha == 2:
            return SparsemaxLoss().predict(theta)

        if self.alpha == np.inf:
            return PerceptronLoss().predict(theta)

        # General case.
        am1 = self.alpha - 1

        def omega_p(t):
            return (t ** am1 - 1.) / am1

        def omega_p_inv(s):
            return (1 + am1 * s) ** (1. / am1)

        return _bisection(theta, omega_p, omega_p_inv, self.max_iter, self.tol)

    def Omega(self, p):
        p = np.array(p)

        if self.alpha == 1:
            # We need to handle the limit case to avoid division by zero.
            return LogisticLoss().Omega(p)

        scale = self.alpha * (self.alpha - 1)
        return (np.sum((p ** self.alpha), axis=1) - 1.) / scale