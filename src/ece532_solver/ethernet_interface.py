from io import BufferedWriter
import numpy as np
import struct
import socket
import _thread
from tqdm import tqdm

HOST = "192.168.1.11"  # IP address of External Computer
PORT = 22  # The port used by this TCP server


def open_new_client(tableau, connection, BUFFER_SIZE=1446, RECV_BUFFER_SIZE=64):
    print("CONNECTION")

    # Loop until connection closed
    while True:
        # Receive and handle command
        data = connection.recv(BUFFER_SIZE)

        if data != b"READY!":
            print("Error: unknown packet from client")
            break

        #######################################################################################
        ########################### SEND TABLEAU OVER CODE BELOW ##############################
        #######################################################################################

        # Start the transfer
        print("PACKET RECEIVED. DATA: READY! Starting transfer...\n")

        # Send row and column info first
        num_rows, num_cols = tableau.shape
        connection.send(struct.pack("!II", num_rows, num_cols))

        # Send actual matrix next
        # Pack and send the matrix in chunks
        tableau = tableau.flatten()

        # Each 2892 byte wide TCP packet can only hold 723 float32 numbers
        FP_per_packet = int(BUFFER_SIZE / 4)

        len_tableau = len(tableau)


        for i in tqdm(range(0, len_tableau, FP_per_packet), desc="Sending tableau", ascii=True):
            # Slice the flattened matrix to get the current chunk
            chunk = tableau[i : min(i + FP_per_packet, len_tableau)]

            # Pack the chunk using struct.pack and send it over the connection
            packed_chunk = struct.pack(f"!{len(chunk)}f", *chunk)
            connection.send(packed_chunk)

        tableau = tableau.reshape((num_rows, num_cols))

        # Print the elements in the first and last of each row
        # for j in range(0, len(tableau)):
        #     print("ROW {0} BEGINNING: {1}".format(j, tableau[j][0]))
        #     print("ROW {0} END: {1}\n".format(j, tableau[j][-1]))

        # Print the elements in last row
        # print(tableau[-1])
        # print(tableau[0][0:10])

        #######################################################################################
        ######################### RECEIVE TABLEAU BACK CODE BELOW #############################
        #######################################################################################

        # Wait until data is sent back
        num_elements_recvd = 0

        byte_buffer = b""
        # Initialize empty tableau to hold our results
        recvd_tableau = np.array([], dtype=np.float32)

        for i in tqdm(
            range(0, num_cols -1, RECV_BUFFER_SIZE), desc="Receiving results", ascii=True
        ):
            while num_elements_recvd <= i:
                byte_buffer += connection.recv(RECV_BUFFER_SIZE)

                # Process each 4 bytes in buffer
                while len(byte_buffer) >= 4:
                    largest_multiple = len(byte_buffer) // 4

                    floats_rcvd = struct.unpack(
                        f"<{largest_multiple}f",
                        byte_buffer[: largest_multiple * 4],
                    )
                    recvd_tableau = np.append(recvd_tableau, floats_rcvd)
                    byte_buffer = byte_buffer[largest_multiple * 4 :]

                    num_elements_recvd += largest_multiple


        print(recvd_tableau)

    # Close the connection if break from loop
    connection.close()
    print("CONNECTION CLOSED")


def transfer_to_fpga(tableau):
    # Setup the socket
    connection: socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind to an address and port to listen on
    connection.bind((HOST, PORT))
    connection.listen(10)
    print("BEGIN LISTENING ON PORT", PORT)

    # Loop forever, accepting all connections in new thread
    while True:
        new_conn, _ = connection.accept()
        _thread.start_new_thread(
            open_new_client,
            (
                tableau,
                new_conn,
            ),
        )


def transfer_to_simulated_solver(tableau, file_path):
    def solve_model():
        print("SOLVING MODEL...")
        return file_path + "_results"

    class FileConnection:
        def __init__(self) -> None:
            self.file: BufferedWriter = open(file_path, "wb")
            self.result_file = None
            self.signal_ready = True

        def send(self, data):
            self.file.write(data)

        def recv(self, size):
            if self.signal_ready:
                self.signal_ready = False
                return b"READY!"

            if self.result_file is None:
                self.file.close()
                result_file_path = solve_model()
                self.result_file = open(result_file_path, "rb")

            return self.result_file.read(size)

        def close(self):
            self.result_file.close()

    file_connection = FileConnection()
    open_new_client(tableau, file_connection, BUFFER_SIZE=10_000)


if __name__ == "__main__":
    try:

        num_rows = 25
        num_cols = 1034

        # Dummy matrix ~ 94 MiB
        tableau = np.random.rand(num_rows, num_cols)
        tableau = tableau * 2000 - 1000
        tableau = tableau.astype(np.float32)

        print(f"TABLEAU SIZE: {tableau.nbytes:,} bytes\n")
        # transfer_to_fpga(tableau)
        transfer_to_simulated_solver(tableau, "tableau")
    except KeyboardInterrupt:
        pass
