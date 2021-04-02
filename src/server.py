import socket
import os
from thread import *
import struct
import json
import numpy as np
from  Atomic_node_only_lstm import  Atomic_node_only_lstm
from overall_event_get_input import *
from beam_search_2 import parse_arguments

args = parse_arguments()
ServerSocket = socket.socket()
host = args.ip
port = args.port
ThreadCount = 0
try:
    ServerSocket.bind((host, port))
except socket.error as e:
    print(str(e))

print('Waitiing for a Connection..')
ServerSocket.listen(5)

event_net = EDNet()
event_net.load_state_dict(torch.load(args.event_model_path))
if torch.cuda.is_available():
    event_net.cuda()
event_net.eval()

# atomic model
atomic_event_net = Atomic_node_only_lstm()
load_best_checkpoint(atomic_event_net, path=args.atomic_event_path)
if torch.cuda.is_available():
    atomic_event_net.cuda()
atomic_event_net.eval()

data_type = 9
def threaded_client(connection):
    connection.send(str.encode('Welcome to the Server\n'))
    count = 0
    test_seqs = []
    obj_seq_list = []
    seq_id = -1
    temp_count = 0
    while True:
        data = connection.recv(1024)
        if len(data) == 0:
            break
        if count%data_type == 0:
            unpack_data = data.decode()
            if unpack_data == 'result':
                break
            seq_id += 1
            test_seqs.append([])
            test_seqs[seq_id].append(str(unpack_data))
            connection.sendall('clip received'.encode())
            count += 1
            continue
        elif count%data_type == 1:
            
            unpack_data = struct.unpack('<1f', data)
            test_seqs[seq_id].append(unpack_data[0])
            connection.sendall('start id received'.encode())
            count += 1
            continue
        elif count%data_type == 2:
            unpack_data = struct.unpack('<{}f'.format(len(data)/4), data)
            obj_seq_list.append(unpack_data)
            connection.sendall('obj id1 received'.encode())
            count += 1
            continue
        elif count%data_type == 3:
            unpack_data = struct.unpack('<{}f'.format(len(data)/4), data)
            obj_seq_list.append(unpack_data)
            connection.sendall('obj id2 received'.encode())
            count += 1
            continue
        elif count%data_type == 4:
            unpack_data = data.decode().split(' ')
            unpack_data = list(map(str, unpack_data))
            obj_seq_list.append(unpack_data)
            connection.sendall('obj1 type received'.encode())
            count += 1
            continue
        elif count%data_type == 5:
            unpack_data = data.decode().split(' ')
            unpack_data = list(map(str, unpack_data))
            obj_seq_list.append(unpack_data)
            connection.sendall('obj2 type received'.encode())
            count += 1
            continue
        elif count%data_type == 6:
            unpack_data = struct.unpack('<{}f'.format(len(data)/4), data)
            obj_seq_list.append(unpack_data)
            connection.sendall('has obj1 received'.encode())
            count += 1
            continue
        elif count%data_type == 7:
            unpack_data = struct.unpack('<{}f'.format(len(data)/4), data)
            obj_seq_list.append(unpack_data)
            connection.sendall('has obj2 received'.encode())
            count += 1
            continue
        elif count%data_type == 8:
            unpack_data = struct.unpack('<{}f'.format(len(data)/4), data)
            obj_list = unpack_data
            connection.sendall('obj list received'.encode())
            count = 0
        
        c = list(zip(obj_seq_list[0], obj_seq_list[1], obj_seq_list[2], obj_seq_list[3], obj_seq_list[4], obj_seq_list[5]))
        test_seqs[seq_id].append(c)
        test_seqs[seq_id].append(obj_list)

    # print(test_seqs)
    test_set = mydataset_atomic(test_seqs, args)
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_atomic,
                                                batch_size=72, shuffle=False)
    test_results = test(test_loader, atomic_event_net, args)
    input1, input2 = merge_results(test_results)
    input1_pad = np.zeros((1, 50))
    input2_pad = np.zeros((1, 50))
    for i in range(len(input1)):
        input1_pad[0, i] = input1[i]
    for i in range(len(input2)):
        input2_pad[0, i] = input2[i]
    input1s = torch.tensor(input1_pad).float().cuda()
    input2s = torch.tensor(input2_pad).float().cuda()
    outputs = event_net(input1s, input2s)
    outputs = torch.sigmoid(outputs)
    outputs = outputs / torch.sum(outputs)
    outputs = outputs.data.cpu().numpy()
    result = list(outputs[0])
    connection.sendall(struct.pack('<{}f'.format(len(result)), *result))
    connection.close()

while True:
    Client, address = ServerSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    start_new_thread(threaded_client, (Client, ))
    ThreadCount += 1
    print('Thread Number: ' + str(ThreadCount))
ServerSocket.close()