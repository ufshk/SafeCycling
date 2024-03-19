import ftplib
import ipaddress
from os import path
import time
import sys
import warnings
import argparse

import serial
import serial.tools.list_ports

import eBUS as eb
import cv2

# could maybe import * from common but you know
from common.chunk_parser import decode_chunk
from common.connection import init_bottlenose, deinit_bottlenose
from common.draw_bboxes import draw_bounding_boxes
from common.distance_estimator import first_distance_estimator, calibrated_img_dist_estimator
from common.notification import connect_to_arduino, send_command, command_selector


def handle_buffer(pvbuffer, device):
    payload_type = pvbuffer.GetPayloadType()

    if payload_type == eb.PvPayloadTypeImage:
        image = pvbuffer.GetImage()
        image_data = image.GetDataPointer()

        bounding_boxes = decode_chunk(device=device, buffer=pvbuffer, chunk='BoundingBoxes')

        # Draw any bounding boxes found
        if bounding_boxes is not None:
            draw_bounding_boxes(image_data, bounding_boxes)

        # Bottlenose sends as YUV422
        if image.GetPixelType() == eb.PvPixelYUV422_8:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_YUV2BGR_YUY2)

            cv2.imshow("Detections", image_data)

        return bounding_boxes, image_data


def read_calibration(file_name):
    """
    Load the contents of calibration file into a dictionary
    :param file_name:
    :return:
    """
    # TODO
    return dict()


def __set_register(device, regname, regvalue):
    reg = device.GetParameters().Get(regname)
    if not reg:
        return False
    res, regtype = reg.GetType()
    if not res.IsOK():
        return False
    if regtype == eb.PvGenTypeFloat:
        res = reg.SetValue(regvalue)
    elif regtype == eb.PvGenTypeInteger:
        res = reg.SetValue(int(regvalue))
    elif regtype == eb.PvGenTypeCommand:
        if bool(regvalue):
            res = reg.Execute()
    else:
        return False
    return res.IsOK()


def upload_calibration(device, file_name):
    """
    Upload weights file to Bottlenose.
    :param device: The device to upload calibration values to
    :param file_name: The file to upload (do not unpack the .tar file)
    """
    if not path.exists(file_name):
        raise RuntimeError(f"Unable to find calibration file {file_name}")

    kparams = read_calibration(file_name)
    for kname, kvalue in kparams.items():
        if not __set_register(kname, kvalue):
            raise RuntimeError(f"Failed to set [{kname}] on the sensor")

    res = __set_register(device, "saveCalibrationData", 1)

    return res.IsOK()


def upload_weights(device, file_name):
    """
    Upload weights file to Bottlenose.
    :param device: The device to upload weights to
    :param file_name: The file to upload (do not unpack the .tar file)
    """
    if not path.exists(file_name):
        raise RuntimeError(f"Unable to find weights file {file_name}")

    # Get device parameters need to control streaming
    device_params = device.GetParameters()
    weights_update = device_params.Get("EnableWeightsUpdate")
    weights_status = device_params.Get("DNNStatus")
    current_ip = device_params.Get("GevCurrentIPAddress")
    if weights_update is None or weights_status is None:
        raise RuntimeError("Unable to find weights update or status parameters, please update your firmware")

    res, ip_address = current_ip.GetValue()
    if res.IsOK():
        ip_address = str(ipaddress.IPv4Address(ip_address))
    else:
        raise RuntimeError("Unable to find current IP address")

    # Enable weights update
    res, val = weights_update.GetValue()
    if val:
        weights_update.SetValue(False)
        print('Disabling weights update')

    weights_update.SetValue(True)
    while True:
        time.sleep(0.1)
        res, status = weights_status.GetValue()
        print(status)
        if res.IsOK():
            if status.find('FTP running') >= 0:
                break
        else:
            raise RuntimeError(f"Unable to update weights, status: {status}")

    # Upload weights file
    ftp = ftplib.FTP(ip_address)
    ftp.login()
    try:
        ftp.storbinary(f"STORE {path.basename(file_name)}", fp=open(file_name, "rb"))
    except Exception as e:
        # Bottlenose validates the uploaded file and will error out if the format is corrupted
        raise RuntimeError(f"Unable to upload weights file: {e}, possibly corrupted file")
    finally:
        ftp.close()

    valid = False
    for _ in range(100):
        res, status = weights_status.GetValue()
        if res.IsOK():
            if status.find('Loaded') >= 0:
                valid = True
                break
        time.sleep(0.1)
    if not valid:
        raise RuntimeError(f"Unable to load weights, status: {status}")

    # Disable weights update
    weights_update.SetValue(False)


def configure_camera(device, outside=True):
    """
    Configure camera settings, namely resolution and colour profile
    :param outside: outdoors? touching grass? it could happen
    :param device: The device to enable bounding box streaming for
    """
    # set resolution to 1080p (to reduce compute time for nn)
    height = device.GetParameters().Get("Height")
    height.SetValue(720)

    exposure = device.GetParameters().Get("exposure")
    if outside:
        exposure.SetValue(0.5)
    else:
        exposure.SetValue(25)

    gain = device.GetParameters().Get("gain")
    gain.SetValue(1)

    # TODO - potential colour profile work???


def enable_bounding_boxes(device):
    """
    Enable the bounding points chunk data
    :param device: The device to enable bounding box streaming for
    """
    # Get device parameters
    device_params = device.GetParameters()

    # Enable keypoint detection and streaming
    chunk_mode = device_params.Get("ChunkModeActive")
    chunk_mode.SetValue(True)
    chunk_selector = device_params.Get("ChunkSelector")
    chunk_selector.SetValue("BoundingBoxes")
    chunk_enable = device_params.Get("ChunkEnable")
    chunk_enable.SetValue(True)


def configure_model(device, confidence=0.2):
    """
    Configure the AI model
    :param device: device
    :param confidence: Confidence threshold for detections
    """
    dnn_enable = device.GetParameters().Get("DNNEnable")
    dnn_confidence = device.GetParameters().Get("DNNConfidence")
    dnn_debug = device.GetParameters().Get("DNNDrawOnStream")

    if dnn_debug is None or dnn_enable is None or dnn_confidence is None:
        raise RuntimeError("Unable to find DNN debug parameter, please update your firmware")

    # Please use eBusPlayer to find the best parameter of your model
    dnn_enable.SetValue(True)
    dnn_confidence.SetValue(confidence)

    # Set this parameter to true for Bottlenose draw bounding boxes on the stream before transmitting it out
    dnn_debug.SetValue(False)


def run_stream(device, stream, weights_file, calibration_file, args):
    """
    Run the demo
    :param device: The device to stream from
    :param stream: The stream to use for streaming
    :param weights_file: The path to the AI model weights file to upload.
    :param calibration_file: The path to the file to calibrate the camera, undistorting images
    :param args: other parameters for running various code functions are in here
    """
    # Get device parameters need to control streaming
    device_params = device.GetParameters()

    # Map the GenICam AcquisitionStart and AcquisitionStop commands
    start = device_params.Get("AcquisitionStart")
    stop = device_params.Get("AcquisitionStop")

    # Calibration values for undistorted images
    if calibration_file is not None:
        upload_calibration(device, calibration_file)

    # Enable keypoint detection and streaming
    if weights_file is not None:
        upload_weights(device, weights_file)

    # Configure the camera
    configure_camera(device)

    # Enable chunk data transfer for bounding boxes
    enable_bounding_boxes(device)

    # Configure the parameters
    configure_model(device)

    # return bounding box dimensions for calculating distance
    # f = open("bboxdata/log_3m_distance_3ft_over_person.txt", "w")
    # f.write("label, width, height, score\n")

    ser = connect_to_arduino(args.alerts)

    # Enable streaming and send the AcquisitionStart command
    device.StreamEnable()
    start.Execute()

    history = [True, False, False]
    while True:
        # Retrieve next pvbuffer
        result, pvbuffer, operational_result = stream.RetrieveBuffer(1000)
        bboxes, img_data = None, None
        if result.IsOK():
            if operational_result.IsOK():
                # We now have a valid buffer.
                bboxes, img_data = handle_buffer(pvbuffer, device)

                if calibration_file:
                    if args.nndepth:
                        est_dist = None
                    else:
                        est_dist = calibrated_img_dist_estimator(bboxes)
                else:
                    est_dist, _ = first_distance_estimator(bboxes)

                if args.alerts:
                    history = command_selector(est_dist, ser, history)

                if args.data_collection:
                    pass

                if cv2.waitKey(1) & 0xFF != 0xFF:
                    break
            else:
                # Non-OK operational result
                warnings.warn(f"Operational result error. {operational_result.GetCodeString()} "
                              f"({operational_result.GetDescription()})",
                              RuntimeWarning)
            # Re-queue the pvbuffer in the stream object
            stream.QueueBuffer(pvbuffer)
        else:
            # Retrieve pvbuffer failure
            warnings.warn(f"Unable to retrieve buffer. {result.GetCodeString()} ({result.GetDescription()})",
                          RuntimeWarning)

    # Tell the Bottlenose to stop sending images.
    stop.Execute()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="For your SafeCycling needs")

    parser.add_argument("-wf", "--weights-file", type=str)
    parser.add_argument("-cf", "--calibration-file", type=str)
    parser.add_argument("-mac", "--mac-address", type=str)
    parser.add_argument("-a", "--alerts", action="store_true")
    parser.add_argument("-nnd", "--nn-depth", action="store_true")
    parser.add_argument("-data", "--data-collection", action="store_true")

    args = parser.parse_args()

    device, stream, buffers = init_bottlenose(args.mac_address)
    if device is not None:
        run_stream(device, stream, args.weights_file, args.calibration_file, args)

    deinit_bottlenose(device, stream, buffers)
