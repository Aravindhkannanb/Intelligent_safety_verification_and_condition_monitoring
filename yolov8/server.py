import socket

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the server address and port
server_address = ('0.0.0.0', 8888)  # Use 0.0.0.0 to listen on all available interfaces
server_socket.bind(server_address)

# Listen for incoming connections
server_socket.listen(1)
print("Waiting for connection...")

# Accept a connection
client_socket, client_address = server_socket.accept()
print("Connected to:", client_address)

while True:
    # Receive the safety status message from the client
    safety_status = client_socket.recv(1024).decode()

    if not safety_status:
        break

    # Print the received safety status
    print("Safety Status:", safety_status)

# Close the connection
client_socket.close()
server_socket.close()
