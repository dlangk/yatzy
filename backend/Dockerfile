# ---------------------------
# Stage 1: Build
# ---------------------------
FROM ubuntu:22.04 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    gcc \
    libmicrohttpd-dev \
    libjson-c-dev \
    libgomp1 \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory and copy source code
WORKDIR /app
COPY . /app

# Build the project
RUN rm -rf build && mkdir build && cd build && cmake .. && make

# Verify the yatzy binary exists
RUN ls -l /app/build/yatzy

# ---------------------------
# Stage 2: Runtime
# ---------------------------
FROM ubuntu:22.04

# Install minimal runtime dependencies and curl for the healthcheck
RUN apt-get update && apt-get install -y \
    libmicrohttpd-dev \
    libjson-c-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the compiled binary and data directory from the build stage
COPY --from=builder /app/build/yatzy /app/build/yatzy
COPY --from=builder /app/data /app/data

# Expose the port your server listens on
EXPOSE 9000

# Ensure the data directory is writable
RUN chmod -R 777 /app/data && ls -l /app/data

# Set environment variables
ENV YATZY_BASE_PATH=/app
ENV YATZY_DATA_DIR=/app/data

# Add a HEALTHCHECK to verify the server is running
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:9000 || exit 1

# Command to run the server
CMD ["/app/build/yatzy"]