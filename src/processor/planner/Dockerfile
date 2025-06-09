FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies INCLUDING sqlite-devel
RUN yum update -y && \
    yum install -y \
    gcc \
    gcc-c++ \
    pkgconfig \
    openssl-devel \
    sqlite-devel \
    && yum clean all

# Set environment variables for Lambda environment
ENV HOME=/tmp
ENV TMPDIR=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache
ENV XDG_DATA_HOME=/tmp/.local/share
ENV XDG_CONFIG_HOME=/tmp/.config
ENV CHROMA_DB_IMPL=duckdb+parquet
ENV ANONYMIZED_TELEMETRY=False

# Create writable directories
RUN mkdir -p /tmp/.cache /tmp/.local/share /tmp/.config /tmp/chroma_db

# Copy requirements first for better caching
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy function code
# COPY handler.py ${LAMBDA_TASK_ROOT}/
# Copy all application files
COPY . ${LAMBDA_TASK_ROOT}/

# Run the application
CMD ["handler.lambda_handler"]