# Use an ARM64 base image for compatibility with M1 MacBook
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    gcc \
    g++ \
    make \
    libmicrohttpd-dev \
    libjson-c-dev \
    libomp-dev \
    curl \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . /app

# Configure and build the project
RUN cmake . && \
    make

# Expose a port if needed (e.g., for HTTP services)
EXPOSE 8080

# Specify the command to run the built executable
CMD ["./yatzy_c"]