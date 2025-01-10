# Use an official image with CMake and GCC
FROM ubuntu:22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    gcc \
    libmicrohttpd-dev \
    libjson-c-dev \
    libgomp1 \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the source code and data directory
COPY . /app

# Build the project
RUN rm -rf build && mkdir build && cd build && cmake .. && make

# Verify the yatzy binary exists
RUN ls -l /app/build/yatzy

# Expose the port the server listens on
EXPOSE 9000

# Ensure the data directory is writable and verify its contents
RUN chmod -R 777 /app/data
RUN ls -l /app/data

# Set environment variables for the dynamic data directory
ENV YATZY_BASE_PATH=/app
ENV YATZY_DATA_DIR=/app/data

# Command to run the server
CMD ["/app/build/yatzy"]