import serial
import serial.tools.list_ports
import warnings


def connect_to_arduino(alerts):
    """
    Connect to Arduino. It's in the name.
    :param alerts: Is the alerts subsystem (i.e. the Arduino) connected or not?
    :return: either None or a serial object
    """
    ser = None
    if alerts:
        arduino_ports = [
            p.device
            for p in serial.tools.list_ports.comports()
            if 'Arduino' in p.description  # may need tweaking to match new Arduinos
        ]

        if not arduino_ports:
            raise IOError("No Arduino found")
        if len(arduino_ports) > 1:
            warnings.warn('Multiple Arduinos found - using the first')

        ser = serial.Serial(arduino_ports[0], 9600)
        send_command(ser, "L")
    return ser


def send_command(ser, msg):
    """
    "Send command to Arduino"
    :param ser:
    :param msg:
    :return:
    """
    cmd = msg.encode('utf-8')
    ser.write(cmd)
    return


def command_selector(dist, ser, history):
    """
    Selects command to send to Arduino UNO based on current estimated range to the closest vehicle and previous history.
    If previous command sent is the same as one that would be sent, no action taken

    :param dist: the estimated distance to the car
    :param ser: Serial object / port to send command over to Arduino
    :param history: array of bools; True when related command was the last one sent, else False
    :return: new history array
    """
    if dist < 5:
        if history[0]:
            pass
        else:
            send_command(ser, "L")
            history = [False] * 3
            history[0] = True
    elif 5 <= dist < 10:
        if history[1]:
            pass
        else:
            send_command(ser, "M")
            history = [False] * 3
            history[1] = True
    else:
        if history[2]:
            pass
        else:
            send_command(ser, "H")
            history = [False] * 3
            history[2] = True

    return history
