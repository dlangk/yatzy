# Use a lightweight Python image as the base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Ensure the entrypoint script is executable
RUN chmod +x /app/entrypoint.sh

# Expose the port for the Python HTTP server
EXPOSE 8090

# Set the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]