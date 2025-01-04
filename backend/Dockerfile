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

# Ensure the build directory is clean, then build the project
RUN rm -rf build && mkdir build && cd build && cmake .. && make

# Verify the yatzy binary exists
RUN ls -l /app/build/yatzy

# Expose the port the server listens on
EXPOSE 9000

# Ensure the data directory is writable (optional)
RUN chmod -R 777 /app/data

# Ensure the data directory is present
RUN ls -l /app/data

# Set the base path used to locate the data directory
ENV YATZY_BASE_PATH=/app

# Command to run the server
CMD ["/app/build/yatzy"]