# Use lightweight python base image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# Force PyTorch to use the CPU-only wheel to keep the image footprint minimal for Hugging Face Spaces
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Remove torch from requirements temporarily to avoid overwriting with CUDA version, or just install the rest
RUN grep -v "torch" requirements.txt > requirements_no_torch.txt && \
    pip install --no-cache-dir -r requirements_no_torch.txt && \
    rm requirements_no_torch.txt

# Copy the rest of the application and set ownership
COPY --chown=user:user . .

# Switch to the non-root user
USER user

# Expose the default Streamlit and Hugging Face Spaces port
EXPOSE 7860

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
