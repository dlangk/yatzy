# Use a lightweight Python image as the base
FROM python:3.10-slim

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Ensure the entrypoint script is executable
RUN chmod +x /app/entrypoint.sh

# Expose the port for the Python HTTP server
EXPOSE 8090

# Add a HEALTHCHECK to verify that the server is responding
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8090 || exit 1

# Set the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]