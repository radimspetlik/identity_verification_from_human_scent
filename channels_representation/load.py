SCAN_LENGHT_SYSTEM_1 = 714800
SCAN_LENGHT_SYSTEM_2 = 715600
SCAN_LENGHT_SYSTEM_3 = 715600
SCAN_LENGHT_SYSTEM_NEW1 = 715943
SCAN_LENGHT_SYSTEM_NEW2 = 736000
SCAN_LENGHT_SYSTEM_NEW3 = 734000
SCAN_LENGHT_SYSTEM_NEW4 = 754400
SCAN_LENGHT_SYSTEM_NEW5 = 670000
SCAN_LENGHT_SYSTEM_NEW6 = 735999


def set_system_1():
    scan_length = SCAN_LENGHT_SYSTEM_1
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 200
    # y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    y_eight_seconds = 1200
    x_eight_seconds = 149
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds


def set_system_2():
    scan_length = SCAN_LENGHT_SYSTEM_2
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 200
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 111
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds


def set_system_3():
    scan_length = SCAN_LENGHT_SYSTEM_3
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 200
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 111
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds


def set_system_new1():
    scan_length = SCAN_LENGHT_SYSTEM_NEW1
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 201
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 111
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds


def set_system_new2():
    scan_length = SCAN_LENGHT_SYSTEM_NEW2
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 217
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 111
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds

def set_system_weird():
    scan_length = SCAN_LENGHT_SYSTEM_2
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 0
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 261
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds

def set_system_new3():
    scan_length = SCAN_LENGHT_SYSTEM_NEW3
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 217
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 111
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds


def set_system_new4():
    scan_length = SCAN_LENGHT_SYSTEM_NEW4
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 234
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 111
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds


def set_system_new5():
    scan_length = SCAN_LENGHT_SYSTEM_NEW5
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 201
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 111
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds


def set_system_new6():
    scan_length = SCAN_LENGHT_SYSTEM_NEW6
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 217
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 111
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds
