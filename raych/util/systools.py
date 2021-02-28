from contextlib import suppress
import os
import subprocess
import time


def is_windows():
    "是否 windows 系统"
    return os.name == 'nt'


def kill_gracefully(process, timeout=2):
    "关闭进程"
    try:
        with suppress(ProcessLookupError):
            process.terminate()
        stdout, stderr = process.communicate(timeout=timeout)

    except subprocess.TimeoutExpired:
        _, stdout, stderr = kill_hard(process)

    return process.returncode, stdout, stderr


SIGINFO = 29


def kill_hard(process):
    "Kill the specified process immediately using SIGKILL."
    with suppress(ProcessLookupError):
        if not is_windows():
            # this assumes a debug handler has been registered for SIGINFO
            process.send_signal(SIGINFO)
            time.sleep(1)  # give the logger a chance to write out debug info
        process.kill()

    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def run_command_subprocess(cmd, *args, **kwargs):
    """
    执行linux命令
    比如：run_command_subprocess(['rm', '-rf', new_temp_path])，函数内部 join 成字符串

    *args, **kwargs: 为 subprocess.Popen 函数可设置的参数：
        Arguments:
            args: A string, 在终端中输入命令相同.
            executable: 比如 '\bin\bash'.
            shell: bool，是否在shell中执行.

    return：
        subprocess.Popen 对象，属性有stdin, stdout, stderr, pid, returncode
    """
    if is_windows():
        cmd_list = ['cmd', '/V', '/C']
        if isinstance(cmd, str):
            cmd_list.append(cmd)
        else:
            cmd_list.extend(cmd)
        cmd = cmd_list
        cmd = " ".join(cmd_list)
    else:
        should_set_pipefail = (kwargs.get('shell') is True and
                               kwargs.get('executabe') in [None, '/bin/bash']
                               and os.path.exists('/bin/bash')
                               and isinstance(cmd, str))
        cmd = " ".join(cmd_list)
        if should_set_pipefail:
            kwargs['executable'] = '/bin/bash'
            cmd = 'set -o pipefail; ' + cmd
    return subprocess.Popen(cmd, *args, **kwargs)


def get_env_variable_set_command(name, value):
    "设置环境变量"
    if is_windows():
        return 'set {}={}&&'.format(name, value)
    else:
        return 'export {}="{}";'.format(name, value)


def is_writeable(path, check_parent=False):
    '''
    Check if a given path is writeable by the current user.
    
    :returns: True or False
    '''
    if os.access(path, os.F_OK) and os.access(path, os.W_OK):
        return True
    
    if os.access(path, os.F_OK) and not os.access(path, os.W_OK):
        return False
    
    if check_parent is False:
        return False
    else:
        parent_dir = os.path.dirname(path)
        if not os.access(parent_dir, os.F_OK):
            return False

    return os.access(parent_dir, os.W_OK)


def is_readable(path):
    '''
    Check if a given path is readable by the current user.

    :returns: True or False
    '''
    if os.access(path, os.F_OK) and os.access(path, os.R_OK):
        return True
    return False



###########################################
### test ###
###########################################

def sys_path():
    "相关函数调用"
    print("sys.path[0] = ", sys.path[0])
    print("sys.argv[0] = ", sys.argv[0])
    print()

    print("__file__ = ", __file__)
    print()

    print("os.path.abspath(__file__) = ", os.path.abspath(__file__))
    print("os.path.realpath(__file__) = ", os.path.realpath(__file__))
    print()

    print("os.path.dirname(os.path.realpath(__file__)) = ", 
        os.path.dirname(os.path.realpath(__file__)))
    print()
    
    print("os.path.split(os.path.realpath(__file__)) = ", 
        os.path.split(os.path.realpath(__file__)))
    print()

    print("os.path.split(os.path.realpath(__file__))[0] = ", 
        os.path.split(os.path.realpath(__file__))[0])
    print()

    print("os.getcwd() = ", os.getcwd())


if __name__ == '__main__':
    # 只在当前 shell 生效，退出就没了
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    print(sys.path)