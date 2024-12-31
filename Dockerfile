FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    wget \
    make \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Set up matplotlib for non-interactive backend
ENV MPLBACKEND=Agg

# Install Python packages
RUN pip install --no-cache-dir \
    yfinance==0.2.33 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    vectorbt==0.26.2 \
    pandas_ta==0.3.14b \
    pandas-datareader==0.10.0 \
    statsmodels==0.14.1 \
    PyWavelets==1.5.0 \
    numba==0.58.1 \
    ta-lib==0.4.28

# Create workspace and output directories
RUN mkdir -p /workspace/outputs

# Set working directory
WORKDIR /workspace

# Keep container running
CMD ["tail", "-f", "/dev/null"] 