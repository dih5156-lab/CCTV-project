"""부모 감독자 종료 시 자기 AI에만 SIGTERM을 전달하는 Linux 실행 경계."""

import ctypes
import os
import signal
import sys


def main():
    expected_parent = int(sys.argv[1])
    command = sys.argv[2:]
    if sys.platform != "linux" or not command or not os.path.isabs(command[0]):
        raise ValueError("an absolute executable on Linux is required")
    # 멀티스레드 부모의 preexec_fn을 쓰지 않고 새 인터프리터에서 syscall을 설정한다.
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(
        ctypes.c_int(1),
        ctypes.c_ulong(signal.SIGTERM),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result:
        raise OSError(ctypes.get_errno(), "cannot arm parent-death signal")
    # fork와 설정 사이 부모가 죽었다면 AI를 시작하지 않는다.
    if os.getppid() != expected_parent:
        raise SystemExit(1)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
