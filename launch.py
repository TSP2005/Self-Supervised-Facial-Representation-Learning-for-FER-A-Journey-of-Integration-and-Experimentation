#!/usr/bin/python3

import os
import sys
import socket
import random
import argparse
import subprocess
import torch


def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _get_rand_port():
    return random.randrange(20000, 60000)


def init_workdir():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher using torchrun')
    parser.add_argument('--launch', type=str, default='main.py',
                        help='Specify launcher script.')
    parser.add_argument('--np', type=int, default=-1,
                        help='number of processes per node.')
    parser.add_argument('--nn', type=int, default=1,
                        help='number of nodes.')
    parser.add_argument('--port', type=int, default=-1,
                        help='master port for communication')
    parser.add_argument('--nr', type=int, default=0, 
                        help='node rank.')
    parser.add_argument('--master_address', '-ma', type=str, default="127.0.0.1")
    parser.add_argument('--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args, other_args = parser.parse_known_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        cmd = f"CUDA_VISIBLE_DEVICES={args.device} "
    else:
        cmd = ""

    init_workdir()
    master_address = args.master_address
    num_processes_per_worker = torch.cuda.device_count() if args.np < 0 else args.np
    num_workers = args.nn
    node_rank = args.nr

    if args.port > 0:
        master_port = args.port
    elif num_workers == 1:
        master_port = _find_free_port()
    else: 
        master_port = _get_rand_port()

    print(f'Start {args.launch} by torchrun with port {master_port}!', flush=True)
    # torchrun handles --local_rank internally, so no additional flag is needed.
    cmd += f'python3 -m torchrun '
    cmd += f'--nproc_per_node={num_processes_per_worker} '
    cmd += f'--nnodes={num_workers} '
    cmd += f'--node_rank={node_rank} '
    cmd += f'--master_addr={master_address} '
    cmd += f'--master_port={master_port} '
    cmd += f'{args.launch}'

    for argv in other_args:
        cmd += f' {argv}'

    with open('./log.txt', 'wb') as f:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        while True:
            text = proc.stdout.readline()
            if not text:
                break
            f.write(text)
            f.flush()
            sys.stdout.buffer.write(text)
            sys.stdout.buffer.flush()
            exit_code = proc.poll()
            if exit_code is not None:
                break
    sys.exit(exit_code)
