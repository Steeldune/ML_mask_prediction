from ctypes import *

get_env = windll.kernel32.GetEnvironmentStringsW()

while True:
    # See GetEnvironmentStringsW https://msdn.microsoft.com/ru-ru/library/windows/desktop/ms683187(v=vs.85).aspx
    value = c_wchar_p(get_env).value  # WCHAR**
    print(value)
    get_env += len(value) * 2 + 2 # 2 because unicode char takes 2 bytes
    if c_char_p(get_env).value == b'': # if we have null after last string -- we are done.
        break