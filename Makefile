# Use Makefile to run jupyter lab for convenience so we can save args (ip, port, allow-root, etc)
jupyter-lab:
	jupyter lab --ip 0.0.0.0 --port 8888 --allow-root