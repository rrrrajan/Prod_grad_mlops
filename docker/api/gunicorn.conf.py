"""
===============================================================================
Gunicorn Configuration
===============================================================================

This configuration file controls how Gunicorn runs the FastAPI application.

Architecture:

                    Client Request
                           │
                           ▼
                  +------------------+
                  |    Gunicorn      |   ← Master Process
                  +------------------+
                     │      │
          ┌──────────┘      └──────────┐
          ▼                            ▼
 +------------------+          +------------------+
 | Uvicorn Worker 1 |          | Uvicorn Worker 2 |
 +------------------+          +------------------+
          │                            │
          └──────────────┬─────────────┘
                         ▼
                    FastAPI Application

Gunicorn manages worker processes while Uvicorn handles ASGI requests.
"""

import os

# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================

# The network interface and port on which Gunicorn listens.
#
# 0.0.0.0
# --------
# Listen on ALL available network interfaces.
#
# This is REQUIRED inside Docker containers because:
#
#    127.0.0.1
#       ↓
#    Only accessible INSIDE the container.
#
#    0.0.0.0
#       ↓
#    Accessible from outside the container
#    through Docker's port mapping.
#
# Port 8000 matches the Dockerfile:
#
#    EXPOSE 8000
#
bind = "0.0.0.0:8000"


# =============================================================================
# WORKER CONFIGURATION
# =============================================================================

# Number of worker processes.
#
# Every worker is an independent Python process.
#
# Example:
#
# Client Requests
#
#      ─────► Worker 1
#      ─────► Worker 2
#
# allowing multiple requests to be processed simultaneously.
#
# The value is read from an environment variable:
#
#     GUNICORN_WORKERS=4
#
# If not supplied, it defaults to 2 workers.
#
# Why not use:
#
#     CPU * 2 + 1 ?
#
# That recommendation was made for traditional Linux servers.
#
# In Docker/Kubernetes, containers usually have CPU limits,
# so using a fixed default (2) is safer and more predictable.
#
workers = 1


# Worker type.
#
# FastAPI is an ASGI application.
#
# Gunicorn itself only knows WSGI.
#
# UvicornWorker acts as an adapter:
#
# Gunicorn
#      ↓
# UvicornWorker
#      ↓
# FastAPI
#
worker_class = "uvicorn.workers.UvicornWorker"


# =============================================================================
# TIMEOUT SETTINGS
# =============================================================================

# Maximum time (seconds) a worker may spend processing
# a request before Gunicorn assumes it is stuck.
#
# Example:
#
# Long prediction request
#
#      0 --------120 sec--------X
#
# Worker is terminated and restarted.
#
# Default = 120 seconds.
#
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))


# Time given to workers to finish processing after receiving
# a shutdown signal.
#
# Useful during:
#
# • Kubernetes rolling updates
# • docker stop
# • deployment restarts
#
# Gunicorn waits 30 seconds before forcefully terminating
# the worker.
#
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))


# Keep an HTTP connection open for additional requests.
#
# Without keepalive:
#
# Client
#   │
#   ├── Request
#   └── Connection Closed
#
#
# With keepalive:
#
# Client
#   │
#   ├── Request
#   ├── Request
#   ├── Request
#   └── Close
#
# This reduces connection overhead.
#
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))


# =============================================================================
# LOGGING
# =============================================================================

# "-" means:
#
# Write access logs to STDOUT.
#
# Docker automatically captures STDOUT.
#
# Kubernetes also collects STDOUT.
#
accesslog = "-"


# "-" means:
#
# Write error logs to STDERR.
#
errorlog = "-"


# Logging level.
#
# Environment variable:
#
# LOG_LEVEL=debug
# LOG_LEVEL=warning
# LOG_LEVEL=error
#
# Default:
#
# info
#
loglevel = os.getenv("LOG_LEVEL", "info")


# Redirect stdout/stderr produced by the application
# into Gunicorn's logging system.
#
# Example:
#
# print("Prediction Started")
#
# will appear in Docker logs.
#
capture_output = True


# =============================================================================
# WORKER RECYCLING
# =============================================================================

# After handling this many requests,
# Gunicorn restarts the worker.
#
# Why?
#
# Long-running Python processes may slowly increase
# memory usage because of:
#
# • memory fragmentation
# • third-party libraries
# • gradual memory leaks
#
# Restarting workers periodically keeps memory stable.
#
max_requests = 1000


# Adds randomness.
#
# Without jitter:
#
# Worker1 restart → 1000 requests
# Worker2 restart → 1000 requests
#
# Both restart together.
#
# With jitter:
#
# Worker1 → 1047
# Worker2 → 1085
#
# Restarts are staggered.
#
max_requests_jitter = 100


# =============================================================================
# APPLICATION PRELOADING
# =============================================================================

# Load the FastAPI application BEFORE creating worker processes.
#
# Without preload:
#
# Worker1 loads model.pkl
# Worker2 loads model.pkl
#
# Every worker performs its own initialization.
#
# With preload:
#
# Master loads:
#
# • FastAPI
# • model.pkl
# • preprocessor.pkl
#
# Then workers are forked.
#
# On Linux, read-only memory pages can be shared
# between workers using Copy-on-Write (CoW),
# which can reduce overall memory usage and
# shorten worker startup time.
#
# Note:
# This optimization is most effective when the application
# doesn't modify large shared objects after startup.
#
preload_app = False


# =============================================================================
# TEMPORARY DIRECTORY
# =============================================================================

# Gunicorn occasionally needs temporary files.
#
# /dev/shm is an in-memory filesystem (RAM).
#
# Benefits:
#
# • Faster than disk
# • Less disk I/O
#
# Only configure it if it exists.
#
# It exists on most Linux containers.
#
if os.path.exists("/dev/shm"):
    worker_tmp_dir = "/dev/shm"


# =============================================================================
# PROCESS NAME
# =============================================================================

# Gives the Gunicorn master process a recognizable name.
#
# Helpful when inspecting processes:
#
# ps -ef
#
# or
#
# top
#
# Example:
#
# customer-churn-api
#
proc_name = "customer-churn-api"
