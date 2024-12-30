# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all necessary files from the host to the container
COPY ./index.html /app/
COPY ./css /app/css
COPY ./js /app/js

# Expose the port the Python HTTP server will run on
EXPOSE 8090

# Command to start the Python HTTP server
CMD ["python3", "-m", "http.server", "8090"]