FROM python:3.11-slim

# --- Proxy (Build-Args) ---
ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    HTTP_PROXY=${http_proxy} \
    HTTPS_PROXY=${https_proxy} \
    NO_PROXY=${no_proxy}



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



ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

#COPY requirements.txt /app/requirements.txt
#RUN pip install --upgrade pip
#RUN pip install -r /app/requirements.txt
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY app /app

RUN mkdir -p /app/logs

EXPOSE 8501

CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
