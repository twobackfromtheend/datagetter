import json
import os

import numpy as np
import replay_getter as rg
from labeller import Labeller

BOOST_PER_SECOND = 90  # boost used per second out of 255


class DataGetter:

    def __init__(self, file_path, verbose=True):
        # parses jsons from RocketLeagueReplayParser

        self.verbose = verbose

        self.file_path = file_path
        with open(file_path, 'r') as f:
            replay = json.load(f)
        self.replay = replay

        self.properties = replay['Properties']
        # self.version = replay['Version']
        self.map = self.properties['MapName']

        self.replay_data = replay['Frames']
        self.last_frame_no = int(self.properties['NumFrames']) - 1

        self.goals = self.get_goals()
        self.last_goal_frame = self.goals[-1]["FrameNumber"]
        self.all_data, self.player_dicts, self.team_dicts = self.get_data()

        # self.save_data()

        # self.frames = Frame.get_frames(self)

        # self.teams, self.players = self.find_players_and_teams()

    def get_goals(self):
        goals = self.properties["Goals"]
        number_of_goals = len(goals)
        # goal_number = 0
        # for goal in goals:
        #     goal_frame = {}
        #     goal_frame["time"] = goal["Time"]
        #     goal_frame['player_name'] = goal["PlayerName"]
        #     goal_frame['player_team'] = goal["PlayerTeam"]

        # find frame where time > time
        goal_number = 0
        frame_number = 0
        for frame in self.replay_data:
            frame_time = frame["Time"]
            # print(goal_number, number_of_goals)
            if frame_time > goals[goal_number]["Time"]:
                goals[goal_number]["FrameNumber"] = frame_number
                goal_number += 1
                if goal_number == number_of_goals:
                    break
            frame_number += 1

        if self.verbose:
            print('Found goals:', goals)
        return goals

    def get_data(self):
        """
        all_data format:
        {player_actor_id: {
            'inputs': [[positions, rotations, velocities, angular_velocities, boost_amt] [...]],
            'outputs': [[throttle, steer, 'pitch', yaw, roll, jump, boost, handbrake]]
        }}
        currently implemented:
            inputs: posx, posy, posz, rotx, roty, rotz, vx, vy, vz, angvx, angy, angvz, boost_amt
            outputs: throttle, steer, handbrake, boost, jump, doublejump, dodge
        """
        all_data = {'ball': {}}
        temp_data = []

        # dictionaries to contain data in frames
        player_dicts = {}  # player_actor_id: player_dict
        team_dicts = {}

        player_car_ids = {}  # player_actor_id: car_actor_id
        car_player_ids = {}  # car_actor_id: player_actor_id
        # car_linked_ids = {}  # car_actor_id: {'player', 'jump, doublejump, '}

        current_actor_ids = []
        current_actors = {}  # id:actor_update

        # stores car_actor_ids to collect data for at each frame
        current_car_ids_to_collect = []

        frames = self.replay_data
        frame_no = 0
        current_goal_number = 0

        # loop through frames
        for frame in frames:
            # don't bother after last goal
            if frame_no > self.last_goal_frame:
                break

            _f_time = frame["Time"]
            _f_delta = frame["Delta"]
            # print(frame_no, _f_time)

            # remove deleted actors
            for deleted_actor_id in frame["DeletedActorIds"]:
                current_actor_ids.remove(deleted_actor_id)
                current_actors.pop(deleted_actor_id)
                if deleted_actor_id in car_player_ids:
                    player_actor_id = car_player_ids[deleted_actor_id]
                    car_player_ids.pop(deleted_actor_id)
                    player_car_ids.pop(player_actor_id)

            # apply actor updates
            for actor_update in frame["ActorUpdates"]:
                actor_id = actor_update["Id"]

                # add if new actor
                if actor_id not in current_actor_ids:
                    current_actor_ids.append(actor_id)
                    current_actors[actor_id] = actor_update
                else:
                    # update stuff in current_actors
                    for _k, _v in actor_update.items():
                        current_actors[actor_id][_k] = _v

            # find players and ball
            for actor_id, actor_data in current_actors.items():
                if actor_data[
                    "TypeName"] == "TAGame.Default__PRI_TA" and "Engine.PlayerReplicationInfo:Team" in actor_data:
                    player_dict = {
                        'name': actor_data["Engine.PlayerReplicationInfo:PlayerName"],
                        'team': actor_data["Engine.PlayerReplicationInfo:Team"],
                        # 'steam_id': actor_data["Engine.PlayerReplicationInfo:UniqueId"]["SteamID64"],
                        # 'player_id': actor_data["Engine.PlayerReplicationInfo:PlayerID"]
                    }
                    if actor_id not in player_dicts:
                        # add new player
                        player_dicts[actor_id] = player_dict
                        if self.verbose:
                            print('Found player: %s (id: %s)' %
                                  (player_dict['name'], actor_id))
                        all_data[actor_id] = {'inputs': {}, 'outputs': {}}
                    else:
                        # update player_dicts
                        for _k, _v in actor_data.items():
                            player_dicts[actor_id][_k] = _v
                elif actor_data["ClassName"] == "TAGame.Team_Soccar_TA":
                    team_dicts[actor_id] = actor_data
                    team_dicts[actor_id]['colour'] = 'blue' if actor_data[
                                                                   "TypeName"] == "Archetypes.Teams.Team0" else 'orange'
                # elif actor_data["TypeName"] == "Archetypes.Ball.Ball_Default":
                #     all_data['ball'] = {}

            # stop data collection after goal
            if frame_no > self.goals[current_goal_number]["FrameNumber"]:
                # set all players to sleeping after goal
                for car_actor_id in car_player_ids:
                    current_actors[car_actor_id][
                        "TAGame.RBActor_TA:ReplicatedRBState"]['Sleeping'] = True
                current_goal_number += 1

            # gather data at this frame
            if _f_delta != 0:
                for actor_id, actor_data in current_actors.items():
                    if actor_data["TypeName"] == "Archetypes.Car.Car_Default":
                        if "Engine.Pawn:PlayerReplicationInfo" in actor_data:
                            if "ActorId" in actor_data["Engine.Pawn:PlayerReplicationInfo"]:
                                player_actor_id = actor_data[
                                    "Engine.Pawn:PlayerReplicationInfo"]["ActorId"]
                                # assign car player links
                                player_car_ids[player_actor_id] = actor_id
                                car_player_ids[actor_id] = player_actor_id

                                RBState = actor_data.get(
                                    "TAGame.RBActor_TA:ReplicatedRBState", {})
                                car_is_driving = actor_data.get(
                                    "TAGame.Vehicle_TA:bDriving", False)
                                car_is_sleeping = RBState.get(
                                    'Sleeping', True)

                                # only collect data if car is driving and not
                                # sleeping
                                if car_is_driving and not car_is_sleeping:
                                    current_car_ids_to_collect.append(actor_id)
                                    # print(actor_id, player_actor_id)

                                    throttle = actor_data.get(
                                        "TAGame.Vehicle_TA:ReplicatedThrottle", 128)
                                    steer = actor_data.get(
                                        "TAGame.Vehicle_TA:ReplicatedSteer", 128)
                                    handbrake = actor_data.get(
                                        "TAGame.Vehicle_TA:bReplicatedHandbrake", False)

                                    data_dict = [
                                        RBState['Position']['X'],
                                        RBState['Position']['Y'],
                                        RBState['Position']['Z'],
                                        RBState['Rotation']['X'],
                                        RBState['Rotation']['Y'],
                                        RBState['Rotation']['Z'],
                                        RBState['LinearVelocity']['X'],
                                        RBState['LinearVelocity']['Y'],
                                        RBState['LinearVelocity']['Z'],
                                        RBState['AngularVelocity']['X'],
                                        RBState['AngularVelocity']['Y'],
                                        RBState['AngularVelocity']['Z'],
                                        # _f_delta,
                                    ]

                                    # save data from here
                                    all_data[player_actor_id][
                                        'inputs'][frame_no] = (data_dict)
                                    all_data[player_actor_id][
                                        'outputs'][frame_no] = ([throttle, steer, handbrake])

                                    # temp_data.append(_f_time)

                    elif actor_data["TypeName"] == "Archetypes.Ball.Ball_Default":
                        RBState = actor_data.get(
                            "TAGame.RBActor_TA:ReplicatedRBState", {})
                        ball_is_sleeping = RBState.get('Sleeping', True)
                        if not ball_is_sleeping:
                            # print('safasf')
                            data_dict = [
                                RBState['Position']['X'],
                                RBState['Position']['Y'],
                                RBState['Position']['Z'],
                                RBState['LinearVelocity']['X'],
                                RBState['LinearVelocity']['Y'],
                                RBState['LinearVelocity']['Z'],
                            ]
                            all_data['ball'][frame_no] = data_dict

                for actor_id, actor_data in current_actors.items():
                    if actor_data["TypeName"] == "Archetypes.CarComponents.CarComponent_Boost":
                        car_actor_id = actor_data.get(
                            "TAGame.CarComponent_TA:Vehicle", None)
                        if car_actor_id is not None:
                            car_actor_id = car_actor_id["ActorId"]

                            if car_actor_id in current_car_ids_to_collect:
                                player_actor_id = car_player_ids[car_actor_id]
                                boost_is_active = actor_data.get(
                                    "TAGame.CarComponent_TA:Active", False)
                                if boost_is_active:
                                    # manually decrease car boost amount (not shown in replay)
                                    # i assume game calculates the decrease
                                    # itself similarly
                                    actor_data[
                                        "TAGame.CarComponent_Boost_TA:ReplicatedBoostAmount"] -= int(
                                        _f_delta * BOOST_PER_SECOND)
                                boost_amount = actor_data[
                                    "TAGame.CarComponent_Boost_TA:ReplicatedBoostAmount"]

                                all_data[player_actor_id][
                                    'inputs'][frame_no].append(boost_amount)
                                all_data[player_actor_id][
                                    'outputs'][frame_no].append(boost_is_active)
                for actor_id, actor_data in current_actors.items():
                    if actor_data["TypeName"] == "Archetypes.CarComponents.CarComponent_Jump":
                        car_actor_id = actor_data.get(
                            "TAGame.CarComponent_TA:Vehicle", None)
                        if car_actor_id is not None:
                            car_actor_id = car_actor_id["ActorId"]

                            if car_actor_id in current_car_ids_to_collect:
                                player_actor_id = car_player_ids[car_actor_id]
                                jump_is_active = actor_data.get(
                                    "TAGame.CarComponent_TA:Active", False)

                                all_data[player_actor_id]['outputs'][
                                    frame_no].append(jump_is_active)
                for actor_id, actor_data in current_actors.items():
                    if actor_data["TypeName"] == "Archetypes.CarComponents.CarComponent_DoubleJump":
                        car_actor_id = actor_data.get(
                            "TAGame.CarComponent_TA:Vehicle", None)
                        if car_actor_id is not None:
                            car_actor_id = car_actor_id["ActorId"]

                            if car_actor_id in current_car_ids_to_collect:
                                player_actor_id = car_player_ids[car_actor_id]
                                double_jump_is_active = actor_data.get(
                                    "TAGame.CarComponent_TA:Active", False)

                                all_data[player_actor_id]['outputs'][
                                    frame_no].append(double_jump_is_active)
                for actor_id, actor_data in current_actors.items():
                    if actor_data["TypeName"] == "Archetypes.CarComponents.CarComponent_Dodge":
                        car_actor_id = actor_data.get(
                            "TAGame.CarComponent_TA:Vehicle", None)
                        if car_actor_id is not None:
                            car_actor_id = car_actor_id["ActorId"]

                            if car_actor_id in current_car_ids_to_collect:
                                player_actor_id = car_player_ids[car_actor_id]
                                dodge_is_active = actor_data.get(
                                    "TAGame.CarComponent_TA:Active", False)

                                all_data[player_actor_id]['outputs'][
                                    frame_no].append(dodge_is_active)

            current_car_ids_to_collect = []  # reset ids for next frame
            frame_no += 1

        # numpy all the arrays
        # for player_actor_id in all_data:
        #     all_data[player_actor_id]['inputs'] = np.array(
        #         all_data[player_actor_id]['inputs'], dtype=np.float64)
        #     all_data[player_actor_id]['outputs'] = np.array(
        #         all_data[player_actor_id]['outputs'], dtype=np.float64)

        # print(all_data[player_actor_id]['outputs'].shape)
        # print(all_data)

        # with open('data.json', 'w') as f:
        #     json.dump(all_data, f)
        # with open('tdata.json', 'w') as f:
        #     json.dump(team_dicts, f)

        # print random stuff
        # print(temp_data)
        # x = np.insert(all_data[player_actor_id]['outputs'], 0, np.array(temp_data), axis=1)
        # print(x)
        # csv_header = 'time,throttle,steer,handbrake,boost,jump,doublejump,dodge'
        # np.savetxt("foo.csv", x, delimiter=",", header=csv_header, comments='')

        return all_data, player_dicts, team_dicts

    def save_data(self):
        # player_name: input
        input_arrays = {}
        output_arrays = {}
        for player_actor_id in self.all_data:
            # print(self.player_dicts)
            player_name = self.player_dicts[player_actor_id][
                "Engine.PlayerReplicationInfo:PlayerName"]
            input_arrays[player_name] = self.all_data[
                player_actor_id]['inputs']
            output_arrays[player_name] = self.all_data[
                player_actor_id]['outputs']

            print('Saving data for player: %s' % player_name)
            print('Inputs: %s, Outputs: %s' % (input_arrays[
                                                   player_name].shape, output_arrays[player_name].shape))
            print('Averages:')
            print(np.mean(input_arrays[player_name], axis=0))
            print(np.mean(output_arrays[player_name], axis=0))

        np.savez(os.path.join(os.cwd(), 'input'), **input_arrays)
        np.savez(os.path.join(os.cwd(), 'output'), **output_arrays)


def save_numpy_data_for_dir(replays=None, double=True, label_pitch_roll=True):
    data_dir = rg.get_replay_files(replays=replays)
    replays = rg.find_jsons_in_dir(data_dir)
    replay_count = len(replays)
    print("JSONs found: %s" % replay_count)

    if double:
        if label_pitch_roll:
            from labeller import Labeller
            labeller = Labeller()
        else:
            labeller = None
        all_data = {}
        i = 0
        _parse = False
        for replay in replays:
            # if not _parse:
            #     if os.path.basename(replay) == "FB15A7EE470652451583CA996A59BC18.json":
            #         _parse = True
            #     continue
            try:
                data = DataGetter(replay, verbose=False)
                inputs, outputs = get_double_input_and_output_data(data, labeller)
            except:
                print("\nCould not parse replay %s" % os.path.basename(replay))
                continue

            all_data['%s_inputs' % i] = inputs
            all_data['%s_outputs' % i] = outputs
            print("Parsed replay %s" % os.path.basename(replay))
            i += 1

        save_path = os.path.join(data_dir, 'np_replays2.npz')
        np.savez(save_path, **all_data)
        print('\nSaved %s replays to %s' % (int(len(all_data) / 2), save_path))
    else:
        all_data = {}
        while True:
            inputs, outputs = self.get_input_and_output_data()

            outputs_list = ['throttle', 'steer', 'pitch',
                            'yaw', 'roll', 'jump', 'boost', 'handbrake']
            outputs = np.array(list(outputs['o_%s' % i] for i in outputs_list))

            all_data['%s_inputs' % self.generator_i] = inputs
            all_data['%s_outputs' % self.generator_i] = np.transpose(outputs)

            self.generator_i += 1
            if self.generator_i >= self.replay_count:
                break

        save_path = os.path.join(self.data_dir, 'np_replays.npz')
        np.savez(save_path, **all_data)
        print('Saved %s replays to %s' % (int(len(all_data) / 2), save_path))


def get_input_and_output_data(replay_json):
    try:
        data = DataGetter(replay_json, verbose=False)
        parsed_json = True
    except:
        print("\nCould not parse replay %s" %
              os.path.basename(replay_json))
        return

    inputs = []
    outputs = []
    for player_actor_id, player_actor_data in data.player_dicts.items():
        player_team_actor_id = player_actor_data["team"]["ActorId"]
        player_team_colour = data.team_dicts[
            player_team_actor_id]["colour"]
        # player_name = player_actor_data[
        #     "Engine.PlayerReplicationInfo:PlayerName"]
        # print(player_team_colour, player_name)

        # inputs: posx, posy, posz, rotx, roty, rotz, vx, vy, vz, angvx, angy, angvz, boost_amt
        # outputs: throttle, steer, handbrake, boost, jump, doublejump,
        # dodge
        player_inputs = data.all_data[player_actor_id]["inputs"]
        player_outputs = data.all_data[player_actor_id]["outputs"]
        # parsed inputs: posx, posy, posz, rotx, roty, rotz, vx, vy, vz, angvx, angy, angvz, boost_amt, ballx, bally, ballz, ballvx, ballvy, ballvz
        # parse outputs: throttle, steer, pitch, yaw, roll, jump, boost,
        # handbrake
        ball_data = data.all_data["ball"]
        player_parsed_inputs = []
        player_parsed_outputs = []

        # print(player_actor_id, 'Player:')

        for frame_no in player_inputs:
            # print(frame_no)
            if frame_no in ball_data:
                player_frame_input = player_inputs[
                                         frame_no] + ball_data[frame_no]
                try:
                    throttle, steer, handbrake, boost, jump, doublejump, dodge = player_outputs[
                        frame_no]
                except ValueError:
                    # raise Exception(replay_json, player_actor_id, frame_no)
                    continue
                yaw = steer
                pitch, roll = 0, 0
                jump = jump or doublejump or dodge
                player_frame_output = [
                    throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
                player_parsed_inputs.append(player_frame_input)
                player_parsed_outputs.append(player_frame_output)

        player_parsed_inputs = np.array(player_parsed_inputs)
        player_parsed_outputs = np.array(
            player_parsed_outputs, dtype=np.float32)
        # print(player_parsed_inputs.shape, player_parsed_outputs.shape)

        # change replay rotations to rlbot rotations
        player_parsed_inputs[:, 3:6] *= 32768.
        # print('\n\nimax', np.amax(player_parsed_inputs, axis=0))
        # print('imin', np.amin(player_parsed_inputs, axis=0))

        # change throttle, steer, yaw from replay 0-255 to rlbot -1 to 1
        player_parsed_outputs[:, [0, 1, 3]] = (
                                                      player_parsed_outputs[:, [0, 1, 3]] - 128) / 128.
        # print(player_parsed_outputs.dtype)
        idx = player_parsed_outputs[:, [0, 1, 3]] > 0
        player_parsed_outputs[:, [0, 1, 3]] = np.multiply(
            player_parsed_outputs[:, [0, 1, 3]], np.where(idx, 128 / 127, 1))
        # print('omax', np.amax(player_parsed_outputs, axis=0))
        # print('omin', np.amin(player_parsed_outputs, axis=0))

        player_parsed_inputs, player_parsed_outputs = self.labeller.generate_pitch_and_roll_for_bot(
            player_parsed_inputs, player_parsed_outputs)

        # turn yaw to 0 if roll > 0.2 and roll to 0 otherwise
        player_parsed_outputs[:, 3][
            np.fabs(player_parsed_outputs[:, 4]) > 0.1] = 0

        if player_team_colour == 'orange':
            # negate x, y, vx, vy, avx, avy and ballx, bally, ballvx,
            # ballvy
            player_parsed_inputs[
            :, [0, 1, 6, 7, 9, 10, 13, 14, 16, 17]] *= -1
            # add 180 degrees to yaw
            yaw_slicer = player_parsed_inputs[:, 5] < 0
            player_parsed_inputs[:, 5][yaw_slicer] += 32768
            player_parsed_inputs[:, 5][~yaw_slicer] -= 32768
            # continue
        inputs.append(player_parsed_inputs)
        outputs.append(player_parsed_outputs)

    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)

    inputs, outputs = self.get_parsed_inputs(inputs, outputs)

    inputs, outputs = self.combine_frame_inputs(inputs, outputs, 3)

    # print(inputs.shape, outputs.shape)
    outputs_list = ['throttle', 'steer', 'pitch',
                    'yaw', 'roll', 'jump', 'boost', 'handbrake']
    outputs = {'o_%s' % outputs_list[i]: outputs[
                                         :, i] for i in range(len(outputs_list))}

    # import pandas as pd

    # _input_df = pd.DataFrame(inputs)
    # input_names = ['posx', 'posy', 'posz', 'rotx', 'roty', 'rotz', 'vx', 'vy', 'vz', 'angvx', 'angy', 'angvz', 'boost_amt', 'ballx',
    #                'bally', 'ballz', 'ballvx', 'ballvy', 'ballvz', 'uballx', 'ubally', 'uballz', 'player_front_direction', 'relative_angle_to_ball']
    # _input_df.columns = input_names + input_names + input_names
    # _output_df = pd.DataFrame(outputs)
    # _df = pd.concat((_input_df, _output_df), axis=1)
    # _df.to_csv("df4.csv")
    # raise Exception("hi")
    return inputs, outputs


def get_double_input_and_output_data(data, labeller=None):
    inputs = []
    outputs = []

    number_of_players = len(data.player_dicts)
    ball_data = data.all_data["ball"]

    # get players
    player_colours = {}  # actor_id: colour
    for player_actor_id, player_actor_data in data.player_dicts.items():
        player_team_actor_id = player_actor_data["team"]["ActorId"]
        player_team_colour = data.team_dicts[player_team_actor_id]["colour"]
        # player_name = player_actor_data["Engine.PlayerReplicationInfo:PlayerName"]
        player_colours[player_actor_id] = player_team_colour

    # loop through all frames
    for frame_no in ball_data:
        # check that all players exist
        if not all(frame_no in data.all_data[player_actor_id]["inputs"] for player_actor_id in player_colours):
            continue

        # add frame data for all players
        frame_inputs = []
        frame_outputs = []
        frame_inputs.append(ball_data[frame_no])
        for player_actor_id in player_colours:
            if frame_no in data.all_data[player_actor_id]["inputs"]:
                # inputs
                player_team_colour = player_colours[player_actor_id]
                player_team_input = [0] if player_team_colour == 'blue' else [1]
                player_frame_input = player_team_input + data.all_data[player_actor_id]["inputs"][
                    frame_no]

                # outputs
                try:
                    throttle, steer, handbrake, boost, jump, doublejump, dodge = \
                        data.all_data[player_actor_id]["outputs"][
                            frame_no]
                except ValueError:
                    continue
                yaw = steer
                pitch, roll = 0, 0
                jump = jump or doublejump or dodge
                player_frame_output = [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

                # player_frame_inputs: colour, posx, posy, posz, rotx, roty, rotz, vx, vy, vz, angvx, angy, angvz, boost_amt
                # player_frame_outputs: throttle, steer, pitch, yaw, roll, jump, boost, handbrake
                player_frame_input = np.array(player_frame_input, dtype=np.float64)
                player_frame_output = np.array(player_frame_output, dtype=np.float64)

                # change replay rotations to rlbot rotations
                player_frame_input[4:7] *= 32768.

                # change throttle, steer, yaw from replay 0-255 to rlbot -1 to 1
                player_frame_output[[0, 1, 3]] = (player_frame_output[[0, 1, 3]] - 128) / 128.
                idx = player_frame_output[[0, 1, 3]] > 0
                player_frame_output[[0, 1, 3]] = np.multiply(
                    player_frame_output[[0, 1, 3]], np.where(idx, 128 / 127, 1))

                frame_inputs.append(player_frame_input)
                frame_outputs.append(player_frame_output)
            else:
                # this code appends 0s if player does not exist
                player_team_colour = player_colours[player_actor_id]
                player_team_input = [0] if player_team_colour == 'blue' else [1]
                default_frame_input = player_team_input + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                default_frame_output = [0, 0, 0, 0, 0, 0, 0, 0]
                frame_inputs.append(default_frame_input)
                frame_outputs.append(default_frame_output)

        inputs.append([_fi for sublist in frame_inputs for _fi in sublist])
        outputs.append([_fo for sublist in frame_outputs for _fo in sublist])

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    if labeller is not None:
        inputs, outputs = labeller.generate_pitch_and_roll_for_datagetter(inputs, outputs, number_of_players)

    return inputs, outputs


def get_parsed_inputs(self, inputs, outputs):
    # inputs: posx, posy, posz, rotx, roty, rotz, vx, vy, vz, angvx, angy,
    # angvz, boost_amt, ballx, bally, ballz, ballvx, ballvy, ballvz

    relative_positions = inputs[:, 13:16] - inputs[:, 0:3]
    rotations = inputs[:, 3:6]
    unrotated_positions = utils.unrotate_positions(
        relative_positions, rotations)

    player_x = inputs[:, 0]
    player_y = inputs[:, 1]
    pitch = inputs[:, 4]
    yaw = inputs[:, 5]
    ball_x = inputs[:, 13]
    ball_y = inputs[:, 14]
    _const = np.pi / 32768
    player_rot1 = np.cos(pitch * _const) * np.cos(yaw * _const)
    player_rot4 = np.cos(pitch * _const) * np.sin(yaw * _const)

    player_front_direction = np.arctan2(player_rot1, player_rot4)
    relative_angle_to_ball = np.arctan2(
        (ball_x - player_x), (ball_y - player_y))

    idx = np.absolute(player_front_direction -
                      relative_angle_to_ball) >= np.pi
    player_front_direction = np.add(
        player_front_direction, np.where(np.logical_and(idx, player_front_direction < 0), 2 * np.pi, 0))
    relative_angle_to_ball = np.add(
        relative_angle_to_ball, np.where(np.logical_and(idx, relative_angle_to_ball < 0), 2 * np.pi, 0))

    parsed_inputs = np.column_stack((inputs, unrotated_positions))
    # parsed_inputs = np.column_stack(
    #     (inputs, unrotated_positions, player_front_direction, relative_angle_to_ball))
    # parsed_inputs = np.column_stack((player_front_direction, relative_angle_to_ball))

    # this filters out stuff for atba
    # row_filter = np.where(np.logical_and(np.absolute(player_front_direction - relative_angle_to_ball) > 0.5, np.absolute(player_front_direction - relative_angle_to_ball) < 2.5))[0]
    # print(row_filter)
    # parsed_inputs = parsed_inputs[row_filter, :]
    # outputs = outputs[row_filter, :]
    # print(parsed_inputs.shape, outputs.shape)
    return parsed_inputs, outputs


def combine_frame_inputs(self, inputs, outputs, frames_to_combine):
    input_arrays = []
    for i in range(frames_to_combine):
        if i == 0:
            # first
            input_arrays.append(inputs[:-(frames_to_combine - 1)])
        elif i == frames_to_combine - 1:
            # last
            input_arrays.append(inputs[i:])
        else:
            input_arrays.append(inputs[i:(i + 1 - frames_to_combine)])

    inputs = np.concatenate(input_arrays, axis=1)
    outputs = outputs[(frames_to_combine - 1):]
    # print(inputs.shape, outputs.shape)
    return inputs, outputs


if __name__ == '__main__':
    # DataGetter('8E6267AC486C08F964D4578237950F3D.json')  # throttle

    # jump, jump djump, jump flip R, jump flip Forward
    # DataGetter('888242E34566C9B1587C9785D7113BBB.json')

    # drift left, jump roll right, ... jump yaw right, jump roll left
    # DataGetter('CF71FEBB4E6BAF29140F42AB83E81A63.json')

    # RLCS
    # DataGetter('667494A14FCEA73BEA60D2948664E94A.json')

    # atba
    # DataGetter('64D58FA2472DA65691B33F9D6E6DE5B7.json')

    save_numpy_data_for_dir(replays=0)
