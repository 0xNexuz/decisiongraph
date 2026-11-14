# DecisionGraph: HydraDB (open-source engine, built from source) + the Python app, in one container.

# ---- 1. build the HydraDB graph-node from github.com/hydra-db/hydradb ----
FROM ubuntu:24.04 AS hydradb
ARG HYDRADB_REF=916637477f3f34766b4905b270bd4292109d77e6
ARG RUST_VERSION=1.99.0
ENV DEBIAN_FRONTEND=noninteractive CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup PATH=/usr/local/cargo/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates clang cmake curl git libclang-dev libcypher-parser-dev libgraphblas-dev pkg-config \
 && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION} \
 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/hydra-db/hydradb.git /src && cd /src && git checkout ${HYDRADB_REF}
WORKDIR /src
RUN bash scripts/install_lagraph.sh /opt/lagraph
ENV LAGRAPH_LIB_DIR=/opt/lagraph/lib
RUN cargo build --locked --release --features server-runtime --bin graph-node && strip target/release/graph-node

# ---- 2. runtime ----
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl libcypher-parser-dev libgraphblas-dev python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv
COPY requirements.txt /app/requirements.txt
RUN /venv/bin/pip install --no-cache-dir -r /app/requirements.txt
COPY --from=hydradb /src/target/release/graph-node /usr/local/bin/graph-node
COPY app /app/app
COPY web /app/web
COPY data /app/data
COPY models /app/models
COPY start.sh /app/start.sh
WORKDIR /app
EXPOSE 10000
CMD ["bash", "/app/start.sh"]
