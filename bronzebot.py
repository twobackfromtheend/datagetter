import os
import numpy as np
import time

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, LeakyReLU, PReLU
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from keras.utils import plot_model

import utils

from datagetter import DataGetter

import replay_getter as rg



class Saver(Callback):

    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if self.model.bot.save_weights:
            self.model.save_weights(self.model.bot.weights_file_name)
            print("\nSaved model weights (Saver callback)")
        if self.model.bot.save_model:
            self.model.save(self.model.bot.model_file_name + '.h5')
        # self.model.bot.test_on_data()

    def on_epoch_begin(self, epoch, logs={}):
        print("\n")


class BronzeBot:
    model_file_name = 'atbabot256'
    model_activation = 'elu'
    kernel_regularizer = regularizers.l1(1e-6)
    output_names = {'boolean_output': 'bool', 'other_output': 'range'}

    def __init__(self, use_saved_weights=False, save_weights=False, save_model=True, log=False, single_output=None):
        self.log = log
        self.use_saved_weights = use_saved_weights
        self.save_weights = save_weights
        self.save_model = save_model

        self.single_output = single_output

        if single_output:
            # input_dim = 35 for 1v1, 87 for 3v3  (13 per player)
            self.model = self.generate_single_output_model(single_output, input_dim=87)
            self.model.bot = self

            if use_saved_weights or save_weights:
                self.weights_file_name = self.model_file_name + single_output + '_weights.h5'
                if use_saved_weights:
                    self.load_weights()
        else:
            # input_dim = 35 for 1v1, 87 for 3v3  (13 per player)
            self.model = self.generate_model(input_dim=87)
            self.model.bot = self

            if use_saved_weights or save_weights:
                self.weights_file_name = self.model_file_name + '_weights.h5'
                if use_saved_weights:
                    self.load_weights()

        self.data_dir = rg.get_replay_files(replays=0)

    def setup_for_training(self):
        # self.labeller = Labeller(use_saved_weights=True)
        # plot_model(self.labeller.model, to_file='2.png')
        # plot_model(self.model, to_file='1.png')
        # self.labeller.test_on_data()

        npz_file = np.load(os.path.join(self.data_dir, 'np_replays2.npz'))
        self.replays = list(npz_file[x] for x in npz_file.files)
        print('Files in npz:', npz_file.files)
        # for i in self.replays:
        #     print(i.shape)

        self.replay_count = len(self.replays)
        print("Replay data arrays found: %s" % int(self.replay_count / 2))

        self.generator_i = 0

    def load_weights(self):
        if os.path.isfile(self.weights_file_name):
            self.model.load_weights(self.weights_file_name, by_name=True)
            print('\n\nLoaded model weights from %s' % self.weights_file_name)
        else:
            print('\n\nCannot load weights: File %s does not exist.' %
                  self.weights_file_name)
            print('Continuing with default weights')

    def generate_model(self, input_dim, outputs=1, shared_hidden_layers=0, nodes=256, extra_hidden_layers=2,
                       extra_hidden_layer_nodes=8):
        """Generates and returns Keras model given input dim, outputs, hidden_layers, and nodes"""
        inputs = Input(shape=(input_dim,))
        x = inputs
        for hidden_layer_i in range(1, shared_hidden_layers + 1):
            x = Dense(nodes, activation=self.model_activation, kernel_regularizer=self.kernel_regularizer,
                      name='hidden_layer_%s' %
                           hidden_layer_i)(x)
            x = Dropout(0.4)(x)

        shared_output = x

        outputs_list = {'boolean': ['jump', 'boost', 'handbrake'], 'other': [
            'throttle', 'steer', 'pitch', 'yaw', 'roll']}
        outputs = []
        for _output_type, _output_type_list in outputs_list.items():
            for output_name in _output_type_list:
                x = shared_output
                for hidden_layer_i in range(1, extra_hidden_layers + 1):
                    x = Dense(extra_hidden_layer_nodes, activation=self.model_activation,
                              kernel_regularizer=self.kernel_regularizer,
                              name='hidden_layer_%s_%s' % (output_name, hidden_layer_i))(x)
                    x = Dropout(0.4)(x)

                if _output_type == 'boolean':
                    activation = 'sigmoid'
                else:
                    activation = 'tanh'
                _output = Dense(1, activation=activation,
                                name='o_%s' % output_name)(x)
                outputs.append(_output)

        model = Model(inputs=inputs, outputs=outputs)

        loss = {}
        loss_weights = {}
        for _output_type, _output_type_list in outputs_list.items():
            for output_name in _output_type_list:
                loss[
                    'o_%s' % output_name] = 'binary_crossentropy' if _output_type == 'boolean' else 'mean_absolute_error'
                loss_weights['o_%s' %
                             output_name] = 0.01 if _output_type == 'boolean' else 0.1

        loss_weights['o_steer'] *= 20
        loss_weights['o_boost'] *= 10
        loss_weights['o_throttle'] *= 20
        loss_weights['o_jump'] *= 20
        # loss_weights['o_pitch'] *= 3
        # loss_weights['o_pitch'] *= 0.001
        # loss_weights['o_yaw'] *= 0.001
        # loss_weights['o_roll'] *= 0.001

        # adam = optimizers.Adam(lr=0.01)
        model.compile(optimizer='adam', loss=loss, loss_weights=loss_weights)

        return model

    def generate_single_output_model(self, single_output, input_dim, nodes=4, hidden_layers=2):

        model = Sequential()
        model.add(Dense(nodes, input_dim=input_dim))

        for hidden_layer_i in range(1, hidden_layers + 1):
            model.add(Dense(nodes, activation=self.model_activation, kernel_regularizer=self.kernel_regularizer,
                      name='hidden_layer_%s' % hidden_layer_i))
            model.add(Dropout(0.4))

        outputs_list = {'boolean': ['jump', 'boost', 'handbrake'], 'other': [
            'throttle', 'steer', 'pitch', 'yaw', 'roll']}
        _booleans = ['jump', 'boost', 'handbrake']
        _others = ['throttle', 'steer', 'pitch', 'yaw', 'roll']

        if single_output in _booleans:
            model.add(Dense(1, activation='sigmoid', name='o_%s' % single_output))
            model.compile(optimizer='adam', loss='binary_crossentropy')
        elif single_output in _others:
            model.add(Dense(1, activation='tanh', name='o_%s' % single_output))
            model.compile(optimizer='adam', loss='mean_squared_error')
        else:
            raise Exception("Training unknown single output: %s" % single_output)

        return model

    def train_model_using_generator(self, epochs=2000, steps_per_epoch=100):

        early_stopping = EarlyStopping(monitor='loss', patience=500)
        saver = Saver()
        callbacks = [early_stopping, saver]
        if self.log:
            log_dir = "logs/{}".format(time.strftime("%d-%m %H%M%S",
                                                     time.gmtime()))
            tensorboard = TensorBoard(
                write_graph=False, write_images=False, log_dir=log_dir, histogram_freq=10)
            callbacks.append(tensorboard)
            print("Saving TensorBoard logs to %s" % log_dir)

        validation_data = next(self.generator())
        # print(validation_data)
        self.generator_i = 0

        self.model.fit_generator(self.generator(
        ), steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=validation_data, callbacks=callbacks)

        if self.save_weights:
            self.model.save_weights(self.weights_file_name)

    def generator(self):
        # find next data
        while True:
            inputs = self.replays[self.generator_i]
            outputs = self.replays[self.generator_i + 1]

            # find number of players
            # yield for each player

            _inputs, _outputs = self.get_inputs_from_npz(inputs, outputs)

            for i in range(len(_inputs)):
                if self.single_output:
                    yield _inputs[i], _outputs[i]['o_%s' % self.single_output]
                else:
                    yield _inputs[i], _outputs[i]

            self.generator_i += 2
            if self.generator_i >= self.replay_count:
                self.generator_i = 0

    def get_inputs_from_npz(self, inputs, outputs):
        """Get player training data from inputs and outputs. Returns a list of inputs and a list of outputs"""
        # input format: ball (6), player0(14), player1 (14) ...
        # ballx, bally, ballz, ballvx, ballvy, ballvz,
        # colour, posx, posy, posz, rotx, roty, rotz, vx, vy, vz, angvx, angy, angvz, boost_amt
        # output format: player0 (8), player1 (8) ..
        # throttle, steer, pitch, yaw, roll, jump, boost, handbrake

        # index offsets
        inputs_ball_offset = 6
        inputs_player_offset = 14
        outputs_player_offset = 8

        _inputs = []
        _outputs = []

        ball_inputs = inputs[:, :inputs_ball_offset]

        number_of_players_in_replay = int(outputs.shape[1] / 8)
        for i in range(number_of_players_in_replay):
            player_i_offset = inputs_ball_offset + inputs_player_offset * i
            player_o_offset = outputs_player_offset * i
            player_colour = inputs[0, player_i_offset]

            # +1 on left to ignore colour
            _player_inputs = inputs[:, player_i_offset + 1:player_i_offset + inputs_player_offset]
            _player_outputs = outputs[:, player_o_offset: player_o_offset + outputs_player_offset]

            # further parse inputs
            # change orange to blue
            if player_colour == 1:
                _player_inputs = self.change_orange_to_blue(_player_inputs)
                _ball_inputs = self.change_orange_to_blue(ball_inputs, is_ball=True)
            else:
                _ball_inputs = ball_inputs

            # add unrotated ball positions
            relative_positions = _ball_inputs[:, 0:3] - _player_inputs[:, 1:4]
            rotations = _player_inputs[:, 4:7]
            unrotated_positions = utils.unrotate_positions(relative_positions, rotations)

            # add teammate and opponent data
            teammate_data = []
            opponent_data = []
            for j in range(number_of_players_in_replay):
                if i != j:
                    player_j_offset = inputs_ball_offset + inputs_player_offset * j
                    player_j_colour = inputs[0, player_j_offset]
                    player_j_data = inputs[:, player_j_offset + 1:player_j_offset + inputs_player_offset]

                    if player_colour == 1:
                        # if original player was orange, both need swapping no matter this guy's colour
                        # (teammate or opponent does not matter)
                        player_j_data = self.change_orange_to_blue(player_j_data)

                    if player_j_colour == player_colour:
                        teammate_data.append(player_j_data)
                    else:
                        opponent_data.append(player_j_data)

            _player_inputs = np.concatenate(
                [_ball_inputs, _player_inputs, unrotated_positions] + teammate_data + opponent_data, axis=1)

            outputs_list = ['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake']
            _player_outputs = {'o_%s' % outputs_list[i]: _player_outputs[:, i] for i in range(len(outputs_list))}
            _inputs.append(_player_inputs)
            _outputs.append(_player_outputs)

        return _inputs, _outputs

    def change_orange_to_blue(self, inputs, is_ball=False):
        if is_ball:
            # negate ballx, bally, ballvx, ballvy
            _inputs = np.copy(inputs)
            _inputs[:, [0, 1, 3, 4]] *= -1
            return _inputs
        else:
            # negate x, y, vx, vy, avx, avy
            _inputs = np.copy(inputs)
            _inputs[:, [0, 1, 6, 7, 9, 10]] *= -1
            # add 180 degrees to yaw
            yaw_slicer = _inputs[:, 4] < 0
            _inputs[:, 4][yaw_slicer] += 32768
            _inputs[:, 4][~yaw_slicer] -= 32768

            return _inputs


if __name__ == '__main__':
    # bot = BronzeBot(use_saved_weights=True, save_weights=True, log=False)
    bot = BronzeBot(use_saved_weights=False, save_weights=True, log=False)
    # bot = BronzeBot(use_saved_weights=False, save_weights=True, log=False, single_output='steer')
    bot.setup_for_training()
    bot.train_model_using_generator()
