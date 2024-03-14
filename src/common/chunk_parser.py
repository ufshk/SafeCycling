# coding: utf-8
"""
******************************************************************************
*  Copyright 2024 Labforge Inc.                                              *
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
__author__ = ("G. M. Tchamgoue <martin@labforge.ca>",
              "Thomas Reidemeister <thomas@labforge.ca>")
__copyright__ = "Copyright 2024, Labforge Inc."

import warnings

import eBUS as eb
import numpy as np
from collections import namedtuple
import struct
from datetime import datetime, timedelta


def read_chunk_id(device: eb.PvDeviceGEV, chunk_name: str):
    """
    Read a particular chunk ID.
    :param device: the device to read the chunk from
    :param chunk_name: the name of the chunk to read
    :return: Given a chunk_name, returns the associated chunkID if found, return -1 otherwise
    """
    chunk_id = -1
    chunk_selector = device.GetParameters().Get("ChunkSelector")
    if chunk_selector is None:
        warnings.warn("ChunkSelector not found! Please update the device firmware", RuntimeWarning)
    else:
        res, chunk_reg = chunk_selector.GetEntryByName(chunk_name)
        if res.IsOK():
            res, reg_id = chunk_reg.GetValue()
            if res.IsOK():
                chunk_id = reg_id

    return chunk_id


def has_chunk_data(buffer: eb.PvBuffer, chunk_id: int):
    """
    Returns true if the input buffer has a chunk that matches the ID.
    :param buffer: the buffer to check
    :param chunk_id: the chunk ID to look for
    :return: True if the buffer has a chunk that matches the ID, False otherwise
    """
    if not buffer:
        return False

    if not buffer.HasChunks() or chunk_id < 0:
        return False

    for i in range(buffer.GetChunkCount()):
        rs, cid = buffer.GetChunkIDByIndex(i)
        if rs.IsOK() and cid == chunk_id:
            return True

    return False


def decode_chunk_keypoint(data):
    """
    Decode the input buffer as keypoints.
    each keypoint (x:uint16, y:uint16)
    each set of keypoints comes from a designated frame
    fid 0: LEFT_ONLY, 1: RIGHT_ONLY, 2: LEFT_STEREO, 3: RIGHT_STEREO
    """
    if data is None or len(data) == 0:        
        return None, 0

    fields = ['x', 'y']
    Keypoint = namedtuple('Keypoint', fields)

    num_keypoints = int.from_bytes(data[0:2], 'little')
    frame_id = int.from_bytes(data[2:4], 'little')
    
    if num_keypoints <= 0 or num_keypoints > 0xFFFF:        
        return None, 0
    if frame_id not in [0, 1, 2, 3]:        
        return None, 0

    chunkdata = [Keypoint(int.from_bytes(data[i:(i + 2)], 'little'),
                          int.from_bytes(data[(i + 2):(i + 4)], 'little'))
                 for i in range(4, (num_keypoints + 1) * 4, 4)]

    offset = 0
    if frame_id in [2, 3]:
        offset = (num_keypoints + 1) * 4

    return {'fid': frame_id, 'data': chunkdata}, offset


def decode_chunk_descriptor(data):
    """
    decode the input buffer as descriptor.
    each descriptor can be up to 64 bytes long
    each set of descriptor corresponds to a set of keypoints and comes from a designated frame
    fid 0: LEFT_ONLY, 1: RIGHT_ONLY, 2: LEFT_STEREO, 3: RIGHT_STEREO
    """
    if data is None or len(data) == 0:
        return None, 0

    fields = ['fid', 'nbits', 'nbytes', 'num', 'data']
    Descriptor = namedtuple('Descriptors', fields)

    num_descr = int.from_bytes(data[0:2], 'little')
    frame_id = int.from_bytes(data[2:4], 'little')
    if num_descr <= 0 or num_descr > 0xFFFF:
        return None, 0
    if frame_id not in [0, 1, 2, 3]:
        return None, 0

    len_descr = int.from_bytes(data[4:8], 'little')

    nbytes = 1
    while nbytes < len_descr:
        nbytes <<= 1
    nbytes //= 8

    descr_data = [data[i:(i + nbytes)] for i in range(8, num_descr * 64, 64)]
    chunkdata = Descriptor(frame_id, len_descr, nbytes, num_descr, descr_data)

    offset = 0
    if frame_id in [2, 3]:
        offset = (num_descr * 64) + 8

    return chunkdata, offset


def decode_chunk_bbox(data):
    """
    decode the input buffer as bounding boxes.
    each set of boxes comes from a designated frame
    fid 0: LEFT_ONLY, 1: RIGHT_ONLY, 2: LEFT_STEREO, 3: RIGHT_STEREO
    """
    if data is None or len(data) == 0:
        return None, 0

    frame_id = int.from_bytes(data[0:4], 'little')
    num_boxes = int.from_bytes(data[4:8], 'little')
    if num_boxes <= 0 or (num_boxes * 48 + 8) > len(data):
        return None, 0
    if frame_id not in [0, 1, 2, 3]:
        return None, 0

    fields = ['cid', 'score', 'left', 'top', 'right', 'bottom', 'label']
    BBox = namedtuple('BBox', fields)

    chunkdata = []
    for i in range(8, num_boxes * 48, 48):
        cid = int.from_bytes(data[i:i + 4], 'little')
        score = np.frombuffer(data[i + 4:i + 8], dtype=np.float32)[0]
        left = int.from_bytes(data[i + 8:i + 12], 'little')
        top = int.from_bytes(data[i + 12:i + 16], 'little')
        right = int.from_bytes(data[i + 16:i + 20], 'little')
        bottom = int.from_bytes(data[i + 20:i + 24], 'little')
        label = bytearray(data[i + 24:i + 48]).decode('ascii').split('\0')[0]
        box = BBox(cid, score, left, top, right, bottom, label)
        chunkdata.append(box)

    return chunkdata, frame_id


def decode_chunk_matches(data):
    """
    decode the input buffer as matched keypoints.
    the layout tells the structure of the data
    [0,1] = point16, [2,3] point16x8
    0: COORDINATE_ONLY, 1: INDEX_ONLY
    2: COORDINATE_DETAILED, 3: INDEX_DETAILED
    :return Empty list in case of no matches, or a Match type
    """    
    if data is None or len(data) == 0:
        return []
    
    match_fields = ['layout', 'unmatched', 'points']
    Matches = namedtuple('Matches', match_fields)

    count = int.from_bytes(data[0:4], 'little')
    layout = int.from_bytes(data[4:8], 'little')
    unmatched = int.from_bytes(data[8:12], 'little')
    points = []
        
    if 0 <= layout < 2:
        point_fields = ['x', 'y']
        Point = namedtuple('Point', point_fields)
        for i in range(12, 12 + count * 4, 4):
            x = int.from_bytes(data[i:i + 2], 'little')
            y = int.from_bytes(data[i + 2:i + 4], 'little')
            pt = Point(x, y)
            points.append(pt)
    elif 1 < layout < 4:
        point_fields = ['x', 'y', 'x2', 'y2', 'd2', 'd1', 'n2', 'n1']
        PointDetailed = namedtuple('PointDetailed', point_fields)
        for i in range(12, 12 + count * 16, 16):
            x = int.from_bytes(data[i:i + 2], 'little')
            y = int.from_bytes(data[i + 2:i + 4], 'little')
            x2 = int.from_bytes(data[i + 4:i + 6], 'little')
            y2 = int.from_bytes(data[i + 6:i + 8], 'little')
            d2 = int.from_bytes(data[i + 8:i + 10], 'little')
            d1 = int.from_bytes(data[i + 10:i + 12], 'little')
            n2 = int.from_bytes(data[i + 12:i + 14], 'little')
            n1 = int.from_bytes(data[i + 14:i + 16], 'little')
            pt = PointDetailed(x, y, x2, y2, d2, d1, n2, n1)
            points.append(pt)
    else:        
        return []

    chunkdata = Matches(layout, unmatched, points)    
    return chunkdata


def decode_chunk_pointcloud(data):
    """
    decode the input buffer as set of 3D data.
    """
    if data is None or len(data) == 0:
        return None

    count = int.from_bytes(data[0:4], 'little')
    points = []

    Point3D = namedtuple('Point3D', ['x', 'y', 'z'])
    dt = np.dtype(np.float32)
    dt = dt.newbyteorder('<')
    for i in range(4, 4 + count * 12, 12):
        x = np.frombuffer(data[i:i + 4], dtype=dt)
        y = np.frombuffer(data[i + 4:i + 8], dtype=dt)
        z = np.frombuffer(data[i + 8:i + 12], dtype=dt)

        pt = Point3D(x[0], y[0], z[0])
        points.append(pt)

    return points


def decode_chunk_meta(data):
    """
    Decode the input buffer as meta data.
    """
    if data is None or len(data) == 0:
        return None

    fmt = '<QIff'
    expected_size = struct.calcsize(fmt)

    try:
        real_time, count, gain, exposure = struct.unpack(fmt, data[:expected_size])
    except:
        return None

    # Transform real_time to datetime
    seconds = real_time // 1000
    milliseconds = real_time % 1000
    real_date = datetime.utcfromtimestamp(seconds) + timedelta(milliseconds=milliseconds)
    return namedtuple('Meta', ['real_time', 'count', 'gain', 'exposure'])(real_date, count, gain, exposure)


def decode_chunk_data(data: np.ndarray, chunk: str):
    """
    Decode the input data as a BN chunk data.
    Returns the decoded chunk data.
    An empty array is returned is data can't be decoded.
    """

    chunk_data = None
    if chunk == 'FeaturePoints':
        chunk_data = []
        kp, offset = decode_chunk_keypoint(data)
        if kp is not None:
            chunk_data.append(kp)

            if offset > 0:                
                kp2, _ = decode_chunk_keypoint(data[offset:])
                if kp2 is not None:
                    chunk_data.append(kp2)

    elif chunk == 'FeatureDescriptors':
        chunk_data = []
        descr, offset = decode_chunk_descriptor(data)
        if descr is not None:
            chunk_data.append(descr)
            if offset > 0:                
                descr2, _ = decode_chunk_descriptor(data[offset:])
                if descr2 is not None:
                    chunk_data.append(descr2)

    elif chunk == 'BoundingBoxes':
        chunk_data, _ = decode_chunk_bbox(data)

    elif chunk == 'FeatureMatches':
        chunk_data = decode_chunk_matches(data)

    elif chunk == 'SparsePointCloud':
        chunk_data = decode_chunk_pointcloud(data)

    elif chunk == 'FrameInformation':
        chunk_data = decode_chunk_meta(data)

    return chunk_data


def get_chunkdata_by_id(rawdata: np.ndarray, chunk_id: int = 0):
    """
    In case of multipart transmission, returns the buffer attached to each ID.
    """
    chunk_data = []
    if rawdata is None or len(rawdata) == 0 or chunk_id < 0:
        return chunk_data

    pos = len(rawdata) - 4
    while pos >= 0:
        chunk_len = int.from_bytes(rawdata[pos:(pos + 4)], 'big')  # transmitted as big-endian
        if chunk_len > 0 and (pos - 4 - chunk_len) > 0:
            pos -= 4
            chkid = int.from_bytes(rawdata[pos:(pos + 4)], 'big')  # transmitted as big-endian

            pos -= chunk_len
            if chkid == chunk_id:
                chunk_data = rawdata[pos:(pos + chunk_len)]  # transmitted as little-endian
                break

        pos -= 4

    return chunk_data


def decode_chunk(device: eb.PvDeviceGEV, buffer: eb.PvBuffer, chunk: str):
    """
    Decode the chunk data attached to the input buffer.
    Decoding happens only if the chunk corresponds to the requested chunk

    Keypoints:
      returns a list of maximum two (for stereo) keypoint objects.
      Each keypoint object contains all the feature points detected for a single frames
      Each keypoint object has the following fields:

      fid: the frameID with the following description:
        0: LEFT_ONLY, for MONO camera or a stereo transmitting only the left image

        1: RIGHT_ONLY, for a stereo camera transmitting only the right image

        2: LEFT_STEREO, for the left image in a stereo transmission

        3: RIGHT_STEREO, for the left image in a stereo transmission

      data: a list of (x,y) feature points from the corresponding image

    Descriptors:
      returns a list of maximum two (for stereo) descriptor objects.
      Each descriptor object has the following fields:

      fid: the frame ID, same as Keypoint above

      nbits: the number of bits of the AKAZE descriptor, varies from 128 to 486 bits.

      nbytes: the number of bytes used to represent each descriptor, can be up to 64 bytes.
      num: the number of descriptors in the object

      data: a list containing the AKAZE descriptors, each descriptor occupies nbytes and is nbits long. This list
      assumes the order in which the corresponding keypoint data is generated.

    """
    rawdata = None
    payload = buffer.GetPayloadType()

    chunk_id = read_chunk_id(device=device, chunk_name=chunk)
    if payload == eb.PvPayloadTypeImage:
        if has_chunk_data(buffer=buffer, chunk_id=chunk_id):
            rawdata = buffer.GetChunkRawDataByID(chunk_id)

    elif payload == eb.PvPayloadTypeMultiPart:
        if buffer.GetMultiPartContainer().GetPartCount() == 3:
            chkbuffer = buffer.GetMultiPartContainer().GetPart(2).GetChunkData()
            if chkbuffer.HasChunks():
                dataptr = buffer.GetMultiPartContainer().GetPart(2).GetDataPointer()
                rawdata = get_chunkdata_by_id(rawdata=dataptr, chunk_id=chunk_id)
    chunk_data = decode_chunk_data(data=rawdata, chunk=chunk)

    return chunk_data
