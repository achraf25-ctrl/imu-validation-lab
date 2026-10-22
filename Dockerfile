FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dashboard]"
COPY dashboard ./dashboard
COPY tests ./tests
ENV IMULAB_RUNS=/app/runs MPLBACKEND=Agg
EXPOSE 8501
# generate the demo runs once, then serve the dashboard
CMD ["sh", "-c", "python -m imulab demo --out $IMULAB_RUNS && streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=8501"]
