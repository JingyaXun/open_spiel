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
# limitations under the 
"""Example use of the NeuRD algorithm on Kuhn Poker.

This NeuRD implementation does not use an entropy bonus to ensure that the
current joint policy approaches an equilibrium in zero-sum games, but it
tracks the exact tabular average so that the average policy approaches an
equilibrium (assuming the policy networks train well).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import tensorflow.compat.v1 as tf
import tensorflow 
from open_spiel.python.algorithms import neurd_tsallis as neurd

import pyspiel
import time
tf.enable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 1000, "Number of iterations")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_float("alpha", 1.4, "Alpha for Tsallis")
flags.DEFINE_integer("adaptive_alpha", 0, "Whether use adaptive alpha")
flags.DEFINE_integer("increase", 0, "Whether increase alpha or not")
flags.DEFINE_integer("adaptive_policy", 2, "1 for linear, 2 for exp")
flags.DEFINE_float("gamma", 0.99, "param for exp adaptive policy")
flags.DEFINE_float("random_seed", 1, "random seed")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("print_freq", 1 , "How often to print the exploitability")
flags.DEFINE_integer("num_hidden_layers", 1,
                     "The number of hidden layers in the policy model.")
flags.DEFINE_integer("num_hidden_units", 512,
                     "The number of hidden layers in the policy model.")
flags.DEFINE_integer(
    "num_hidden_factors", 8,
    "The number of factors in each hidden layer in the policy model.")
flags.DEFINE_boolean(
    "use_skip_connections", True,
    "Whether or not to use skip connections in the policy model.")
flags.DEFINE_integer("batch_size", 100, "The policy model training batch size.")
flags.DEFINE_float(
    "threshold", 2.,
    "Logits of the policy model will be discouraged from growing beyond "
    "`threshold`.")
flags.DEFINE_float("step_size", 1.0, "Policy model step size.")
flags.DEFINE_boolean(
    "autoencode", False,
    "Whether or not to augment the policy model with outputs that attempt to "
    "reproduce the model inputs. The policy model is updated online so "
    "training with the reproduction error as an auxiliary task helps to keep "
    "the model stable in the absence of an entropy bonus.")


def main(_):
  tensorflow.random.set_random_seed(int(FLAGS.random_seed))
  game = pyspiel.load_game(FLAGS.game,
                           {"players": pyspiel.GameParameter(FLAGS.players)})

  # game = pyspiel.load_game(FLAGS.game)

  models = []
  for _ in range(game.num_players()):
    models.append(
        neurd.DeepNeurdModel(
            game,
            num_hidden_layers=FLAGS.num_hidden_layers,
            num_hidden_units=FLAGS.num_hidden_units,
            num_hidden_factors=FLAGS.num_hidden_factors,
            use_skip_connections=FLAGS.use_skip_connections,
            autoencode=FLAGS.autoencode))

  solver = neurd.CounterfactualNeurdSolver(game, FLAGS.alpha, models)

  def _train(model, data):
    neurd.train(
        model,
        data,
        batch_size=FLAGS.batch_size,
        step_size=FLAGS.step_size,
        alpha=FLAGS.alpha,
        threshold=FLAGS.threshold,
        autoencoder_loss=(tf.compat.v1.losses.huber_loss
                          if FLAGS.autoencode else None))

  # start_time = time.time()
  for i in range(FLAGS.iterations):
        # send i into the function to notify the adaptation of alpha
    
    if FLAGS.adaptive_alpha:
      solver.evaluate_and_update_policy(_train, current_iteration=i, alpha=FLAGS.alpha, increase=FLAGS.increase, gamma=FLAGS.gamma, adaptive_policy=FLAGS.adaptive_policy, total_iteration=FLAGS.iterations)
    else:
      solver.evaluate_and_update_policy(_train, alpha=FLAGS.alpha)
    if i % FLAGS.print_freq == 0:
      conv = pyspiel.exploitability(game, solver.average_policy())
      print("Iteration {} exploitability {}".format(i, conv))
    
  # end_time = time.time()
  # print(end_time - start_time)


if __name__ == "__main__":
  app.run(main)
