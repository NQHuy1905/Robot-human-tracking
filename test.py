from utils.data_transfer import DictionaryReceiver

receiver = DictionaryReceiver()
try:
    receiver.start_listening()
    receiver.accept_connection()

    while True:
        received_dict = receiver.receive_dictionary()
        # if received_dict is None:
        #     print("Client disconnected or timeout reached. Waiting for ne>
        #     receiver.accept_connection()
        # else:
        print(f"Received dictionary: {received_dict}")
except KeyboardInterrupt:
    print("\nReceiver stopped by user.")
finally:
    receiver.close()


