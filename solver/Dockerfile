FROM rust:1.84 AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/yatzy /usr/local/bin/yatzy
COPY --from=builder /app/target/release/yatzy-precompute /usr/local/bin/yatzy-precompute

WORKDIR /app

EXPOSE 9000

ENV YATZY_BASE_PATH=/app

CMD ["yatzy"]
