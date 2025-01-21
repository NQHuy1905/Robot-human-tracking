import socket
import json
import select
import struct

class DictionarySender:
    def __init__(self, host='localhost', port=2011):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        if not self.socket:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")

    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Disconnected from server")

    def send_dictionary(self, dictionary):
        if not self.socket:
            raise ConnectionError("Not connected to server. Call connect() first.")
        
        json_data = json.dumps(dictionary).encode()
        # self.socket.sendall(json_data.encode())
        json_length = struct.pack('!I', len(json_data))
        self.socket.sendall(json_length + json_data)
        print(f"Sent dictionary: {dictionary}")

class DictionaryReceiver:
    def __init__(self, host='localhost', port=2011):
        self.host = host
        self.port = port
        self.socket = None
        self.client_socket = None

    def start_listening(self):
        if not self.socket:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            print(f"Listening for connections on {self.host}:{self.port}")

    def accept_connection(self):
        if not self.socket:
            raise ConnectionError("Server not listening. Call start_listening() first.")
        
        self.client_socket, addr = self.socket.accept()
        print(f"Connected by {addr}")

    def recv_all(self, length):
        """Ensure we receive exactly `length` bytes."""
        data = b''
        while len(data) < length:
            packet = self.client_socket.recv(length - len(data))
            if not packet:
                return None  # Connection closed or error
            data += packet
        return data

    def receive_dictionary(self, timeout=1):
        if not self.client_socket:
            raise ConnectionError("No client connected. Call accept_connection() first.")
        
        # ready = select.select([self.client_socket], [], [], timeout)
        # if ready[0]:
        #     data = self.client_socket.recv(4096)
        #     if data:
        #         try:
        #             received_dict = json.loads(data.decode())
        #             return received_dict
        #         except json.JSONDecodeError:
        #             print("Received data is not a valid JSON. Skipping.")
        #     else:
        #         # Connection closed by client
        #         self.client_socket.close()
        #         self.client_socket = None
        #         return None
        # return None
        ready = select.select([self.client_socket], [], [], timeout)
        if ready[0]:
            # First, receive the length of the incoming message (4 bytes)
            raw_msglen = self.recv_all(4)
            if not raw_msglen:
                return None  # Connection was closed or error
            
            msglen = struct.unpack('!I', raw_msglen)[0]
            
            # Now receive the actual data based on the message length
            data = self.recv_all(msglen)
            if data is None:
                return None  # Connection was closed or error
            
            # Now that the full message is received, decode the JSON
            try:
                received_dict = json.loads(data.decode())
                return received_dict
            except json.JSONDecodeError:
                print("Received data is not a valid JSON. Skipping.")
                return None

    def close(self):
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
        if self.socket:
            self.socket.close()
            self.socket = None
        print("Closed all connections")

if __name__ == "__main__":

    
    sender = DictionarySender()
    sender.connect()

    try:
        while True:
            dict_data = {'Huy':234823490}
            if not dict_data:
                print("Empty dictionary. Exiting.")
                break
            
            sender.send_dictionary(dict_data)
            
            # if input("Send another dictionary? (y/n): ").lower() != 'y':
            #     break
    finally:
        sender.disconnect()