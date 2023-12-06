from pyngrok import ngrok

public_url = ngrok.connect(addr="8080", proto="http")

print("Public URL: ", public_url)
