# Run a Python http server remotely
python -m http.server 8002
# Forward the localhost:9999 to the remote (boryiings@cw-dfw-cs-001-dc-02:8002).
# So we can access the remote http server locally.
ssh -L 9999:localhost:8002 boryiings@cw-dfw-cs-001-dc-02

# Then we can use the local browser to access the remote server by this URL:
http://localhost:9999/
