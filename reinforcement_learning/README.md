# Oxidized AI Infrastructure

A high-performance machine learning monorepo showcasing core AI and data science solution algorithms implemented natively in **Rust**. This workspace utilizes a polyglot architecture, managing real-time simulation loops and cross-language data validation against established, open-source **Python** frameworks over a high-velocity **FFI (PyO3)** boundary.

## 🏗️ Monorepo System Architecture

This repository operates as a unified control center for three decoupled micro-services, isolating language execution models and system runtimes to maximize throughput:

*   **⚡ Reinforcement Learning (`reinforcement_learning/`):** Implements high-speed tabular Q-Learning and Deep Q-Network (DQN) policy updates in pure Rust, managing continuous-space tracking loops via Python's Gymnasium simulator over FFI.

## 🐳 Containerized DevOps Strategy

To prevent language toolchain drift and dependency bloat, this architecture splits development and deployment targets using a single **Multi-Stage Dockerfile** managed via `docker-compose.yml`:

*   **Option B (Development Target):** A heavy, fully isolated compilation sandbox bundling a native Rust 1.85+ compiler, system build tools, and an embedded Python 3.11 environment. This stage runs real-time algorithmic convergence checks and hermetic snapshot testing.
*   **Option A (Production Target):** A microscopic, stripped-down Linux binary container containing **zero Python dependencies or runtime files**. It simply loads fully trained, serialized model weights (`.json` or `.onnx`) into immutable memory slots to serve lightning-fast, sub-millisecond API inference requests.

## 🚀 Getting Started & Task Runner

The workspace includes a unified task runner (`run.sh`) that abstracts infrastructure overhead. Ensure **Docker Desktop** is open and active on your Mac. You do not need to configure any local tools to build or test the code.

Execute these commands directly from your workspace root terminal:

```bash
# 1. Force an absolute, clean assembly build of the container layers
./run.sh build

# 2. Run the complete automated cross-language integration test suite inside the container
./run.sh test

# 3. Optional: Run tests natively on your Mac host if you have configured local paths
./run.sh dev-test
```

## 🧪 Verification Matrix
All algorithms are audited using **Golden File (Snapshot) Testing**. Instead of relying on brittle external runners, our test suites cross-validate native Rust mathematical vectors against audited Python oracles (`stable-baselines3`, `lenskit`, `pgmpy`), asserting numerical convergence boundaries down to a precise floating-point epsilon threshold (`error < 1e-5`).
