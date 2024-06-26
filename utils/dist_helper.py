import deepspeed


def setup():
    deepspeed.init_distributed()
