import os


# Keep matplotlib headless and writable in constrained environments.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
