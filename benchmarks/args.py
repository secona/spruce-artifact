import argparse

INT_MAX = 2**31 - 1


def parse_build_args(operation_choices):
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=operation_choices)
    parser.add_argument("--gpu-arch", default="sm_86")
    args = parser.parse_args()

    return args


def parse_run_args(operation_choices):
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=operation_choices)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    assert 0 <= args.warmup <= INT_MAX
    assert 1 <= args.repeat <= INT_MAX
    assert 0 <= args.device <= INT_MAX

    return args
