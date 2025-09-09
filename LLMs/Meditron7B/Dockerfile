# Lightweight, stable python image
FROM python:3.11-slim

# --- Proxy (Build-Args) ---
ARG USE_PROXY = false
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
# ENV http_proxy=${http_proxy} \
#    https_proxy=${https_proxy} \
#    no_proxy=${no_proxy} \
#    HTTP_PROXY=${http_proxy} \
#    HTTPS_PROXY=${https_proxy} \
#    NO_PROXY=${no_proxy}


# --- CA-Zertifikate ins Image bringen und registrieren ---
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY certs/*.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates

# (manchmal hilfreich für Python/Requests)
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# --- pip-Konfig für Proxy/Trusted-Hosts ---
RUN mkdir -p /etc/pip \
 && printf "[global]\ntrusted-host = pypi.org\n    files.pythonhosted.org\n" > /etc/pip/pip.conf \
 && if [ -n \"$http_proxy\" ]; then printf \"proxy = %s\n\" \"$http_proxy\" >> /etc/pip/pip.conf; fi


# --- Env config ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

#COPY requirements.txt /app/requirements.txt
#RUN pip install --upgrade pip
#RUN pip install -r /app/requirements.txt
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy app folder into container
COPY app /app/app

# Expose port for FastAPI
EXPOSE 8200

# Run FastAPI server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8200", "--reload"]
