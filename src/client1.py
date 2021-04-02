import socket
import struct
import json
import numpy as np

def send_to_server(seq, ip, port):
    ClientSocket = socket.socket()
    host = ip
    port = port

    # print('Waiting for connection')
    try:
        ClientSocket.connect((host, port))
    except socket.error as e:
        print(str(e))

    Response = ClientSocket.recv(1024)
    # print(Response.decode())
    for rec in seq:
        clip, start_id, obj_list_seq, obj_list = rec

        pack_data = clip.encode()
        ClientSocket.send(pack_data)
        Response = ClientSocket.recv(1024)
        # print(Response.decode())

        pack_data = struct.pack('<1f', start_id)
        ClientSocket.send(pack_data)
        Response = ClientSocket.recv(1024)
        # print(Response.decode())

        obj_id1s = []
        obj_id2s = []
        type1s = []
        type2s = []
        has_obj1s = []
        has_obj2s = []
        for i in range(len(obj_list_seq)):
            obj_id1, obj_id2, type1, type2, has_obj1, has_obj2 = obj_list_seq[i]
            obj_id1s.append(obj_id1)
            obj_id2s.append(obj_id2)
            type1s.append(type1)
            type2s.append(type2)
            has_obj1s.append(float(has_obj1))
            has_obj2s.append(float(has_obj2))
        
        pack_data = struct.pack('<{}f'.format(len(obj_list_seq)), *obj_id1s)
        ClientSocket.send(pack_data)
        Response = ClientSocket.recv(1024)
        # print(Response.decode())
        
        pack_data = struct.pack('<{}f'.format(len(obj_list_seq)), *obj_id2s)
        ClientSocket.send(pack_data)
        Response = ClientSocket.recv(1024)
        # print(Response.decode())

        pack_data = ' '.join(type1s)
        ClientSocket.send(pack_data.encode())
        Response = ClientSocket.recv(1024)
        # print(Response.decode())

        pack_data = ' '.join(type2s)
        ClientSocket.send(pack_data.encode())
        Response = ClientSocket.recv(1024)
        # print(Response.decode())

        pack_data = struct.pack('<{}f'.format(len(obj_list_seq)), *has_obj1s)
        ClientSocket.send(pack_data)
        Response = ClientSocket.recv(1024)
        # print(Response.decode())

        pack_data = struct.pack('<{}f'.format(len(obj_list_seq)), *has_obj2s)
        ClientSocket.send(pack_data)
        Response = ClientSocket.recv(1024)
        # print(Response.decode())

        pack_data = struct.pack('<{}f'.format(len(obj_list)), *obj_list)
        ClientSocket.send(pack_data)
        Response = ClientSocket.recv(1024)
        # print(Response.decode())

    ClientSocket.send('result'.encode())
    Response = ClientSocket.recv(1024)
    result = struct.unpack('<{}f'.format(len(Response)/4), Response)
    # print(result)

    ClientSocket.close()
    return result