import os

def get_replay_files(replays=1):
    replays = ['atba', '1s', 'EU'][replays]
    data_dir = os.path.join(os.getcwd(), 'replays', replays)

    return data_dir


def find_jsons_in_dir(_dir):
    files = [os.path.join(_dir, f) for f in os.listdir(
        _dir) if os.path.isfile(os.path.join(_dir, f)) and f.endswith(".json")]
    return files