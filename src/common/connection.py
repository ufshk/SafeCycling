# coding: utf-8
"""
******************************************************************************
*  Copyright 2023 Labforge Inc.                                              *
*                                                                            *
* Licensed under the Apache License, Version 2.0 (the "License");            *
* you may not use this project except in compliance with the License.        *
* You may obtain a copy of the License at                                    *
*                                                                            *
*     http://www.apache.org/licenses/LICENSE-2.0                             *
*                                                                            *
* Unless required by applicable law or agreed to in writing, software        *
* distributed under the License is distributed on an "AS IS" BASIS,          *
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
* See the License for the specific language governing permissions and        *
* limitations under the License.                                             *
******************************************************************************
"""
__author__ = ("Thomas Reidemeister <thomas@labforge.ca>"
              "G. M. Tchamgoue <martin@labforge.ca>")
__copyright__ = "Copyright 2023, Labforge Inc."

import warnings
import eBUS as eb

BUFFER_COUNT = 16
LABFORGE_MAC_RANGE = '8c:1f:64:d0:e'


def find_bottlenose(mac=None):
    """
    Finds an active Bottlenose sensor on the network.
    :param mac: Optional, provide the mac address of the Bottlenose to connect to
    :return: The connection ID of the bottlenose camera, if multiple cameras are found, the first one is returned
    """
    system = eb.PvSystem()
    system.Find()

    # Detect, select Bottlenose.
    device_vector = []
    for i in range(system.GetInterfaceCount()):
        interface = system.GetInterface(i)
        for j in range(interface.GetDeviceCount()):
            device_info = interface.GetDeviceInfo(j)
            if device_info.GetMACAddress().GetUnicode().find(LABFORGE_MAC_RANGE) == 0:
                device_vector.append(device_info)
                if mac is not None and mac == device_info.GetMACAddress().GetUnicode():
                    return device_info.GetConnectionID()

    if len(device_vector) == 0:
        warnings.warn("No Bottlenose camera found!", RuntimeWarning)
        return None

    # Return first Bottlenose found
    return device_vector[0].GetConnectionID()


def connect_to_device(connection_id):
    """
    Establishes a GigE vision device connection.
    :param connection_id: Result of find_bottlenose()
    :return: Device object representing Bottlenose
    """
    # Connect to the GigE Vision or USB3 Vision device
    result, device = eb.PvDevice.CreateAndConnect(connection_id)
    if device is None:
        warnings.warn(f"Unable to connect to device: {result.GetCodeString()} ({result.GetDescription()})", RuntimeWarning)
    return device


def open_stream(connection_ID):
    """
    Instantiates a GigE stream object that can be used to stream images and chunk data from the device.
    :param connection_id: Result of find_bottlenose()
    :return: Stream object representing Bottlenose
    """
    result, stream = eb.PvStream.CreateAndOpen(connection_ID)
    if stream is None:
        warnings.warn(f"Unable to stream from device. {result.GetCodeString()} ({result.GetDescription()})",
                      RuntimeWarning)
    return stream


def configure_stream(device, stream):
    """
    Configures the feasible GigE vision packet size and sets the local host as endpoint for the device.
    :param device: Result of connect_to_device()
    :param stream: Result of open_stream()
    :return: None
    """
    if isinstance(device, eb.PvDeviceGEV):
        # Negotiate packet size
        device.NegotiatePacketSize()
        # Configure device streaming destination
        device.SetStreamDestination(stream.GetLocalIPAddress(), stream.GetLocalPort())


def configure_stream_buffers(device, stream):
    """
    Creates buffers to receive frames and chunkdata from Bottlenose.
    :param device: Result of connect_to_device()
    :param stream: Result of open_stream()
    :return: List of buffers queued as part of the stream to receive frames and chunk data
    """
    buffer_list = []
    # Reading payload size from device
    size = device.GetPayloadSize()

    # Use BUFFER_COUNT or the maximum number of buffers, whichever is smaller
    buffer_count = stream.GetQueuedBufferMaximum()
    if buffer_count > BUFFER_COUNT:
        buffer_count = BUFFER_COUNT

    # Allocate buffers
    for i in range(buffer_count):
        # Create new pvbuffer object
        pvbuffer = eb.PvBuffer()
        # Have the new pvbuffer object allocate payload memory
        pvbuffer.Alloc(size)
        # Add to external list - used to eventually release the buffers
        buffer_list.append(pvbuffer)

    # Queue all buffers in the stream
    for pvbuffer in buffer_list:
        stream.QueueBuffer(pvbuffer)
    return buffer_list


def activate_stereo(device, value = True):
    """
    Enables side-by-side image streaming from Bottlenose Stereo.
    :param device: Result of connect_to_device()
    :param value: True or False
    :return: Success of the operation.
    """

    is_stereo = False
    model = device.GetParameters().Get("DeviceModelName")

    if model:
        res, name = model.GetValue()
        is_stereo = (res.IsOK() and name.endswith('_ST'))

    value &= is_stereo
    multipart = device.GetParameters().Get("GevSCCFGMultiPartEnabled")
    if multipart:
        multipart.SetValue(value)

    return value


def init_bottlenose(mac_address, stereo=False):
    """
    Initializes the Bottlenose sensor for streaming and maps the device and stream objects.
    :param mac_address: MAC address of the Bottlenose to connect to or None to connect to the first Bottlenose found
    :param stereo: True to enable side-by-side image streaming from Bottlenose Stereo (ignored for Bottlenose Mono)
    :return: device, stream, buffers
    """
    connection_ID = find_bottlenose(mac_address)
    if connection_ID:
        device = connect_to_device(connection_ID)
        if device:
            activate_stereo(device, stereo)
            stream = open_stream(connection_ID)
            if stream:
                configure_stream(device, stream)
                buffers = configure_stream_buffers(device, stream)
                return device, stream, buffers
            else:
                device.Close()
                eb.PvStream.Free(device)
                warnings.warn("Unable to stream from device.", RuntimeWarning)
        else:
            warnings.warn("Unable to connect to device.", RuntimeWarning)
    return None, None, None


def deinit_bottlenose(device, stream, buffers):
    """
    Frees up tied up resources in the device and stream object.
    :param device: Result of connect_to_device()
    :param stream: Result of open_stream()
    return None
    """
    if device is None or stream is None:
        return

    # Disable streaming on the Bottlenose
    device.StreamDisable()

    # Abort all buffers from the stream and dequeue
    stream.AbortQueuedBuffers()
    while stream.GetQueuedBufferCount() > 0:
        stream.RetrieveBuffer()

    # Free stream
    stream.Close()
    del buffers
    eb.PvStream.Free(stream)

    # Disconnect the device
    device.Disconnect()
    eb.PvDevice.Free(device)
