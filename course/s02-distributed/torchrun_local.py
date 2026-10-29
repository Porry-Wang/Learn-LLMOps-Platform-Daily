#!/usr/bin/env python3
"""macOS 本机 torchrun 的兼容启动器（环境修复，不是教学内容）。

问题：
    这台 Mac 的 socket.getfqdn() 会返回 `1.0.0.0...ip6.arpa`（::1 的反向名），
    torchrun 的 rendezvous 在解析这个主机名时反复重试，进程看起来像“卡死”。

修复：
    在本进程内把 getfqdn 固定成 localhost，再把命令行参数原样交给
    torch.distributed.run。子进程只继承 torchrun 注入的环境变量，不受影响。

用法（和 torchrun 一样，只是把命令换成这个脚本）：
    ../../.venv/bin/python torchrun_local.py --standalone --nproc_per_node=4 dist.py

生产 Linux 不需要这个 shim，直接用 torchrun。
"""

import runpy
import socket
import sys

# 只影响当前 Python 进程；这是对宿主机 DNS 反向解析异常的局部绕行。
socket.getfqdn = lambda *args, **kwargs: "localhost"

# torch.distributed.run 会读 sys.argv；第一个元素要是程序名。
sys.argv[0] = "torchrun"
runpy.run_module("torch.distributed.run", run_name="__main__")
