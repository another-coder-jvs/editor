# from pyngrok import ngrok
# import os 
# # Replace with your ngrok auth token
# ngrok.kill()
# # ngrok.set_auth_token("3HRjpFMqfHffIlX0oPqHITBQalK_457XLfBzcPqizqijAyV1p")
# # siniy30775@bocably.com :  https://water-unwed-hardwired.ngrok-free.dev/docs : main api 
# # ngrok.set_auth_token("3IPM6QOpeR1pwv7KGytzu2iQa4c_3ZPPb97NviTrr4HFLg8H2")

# ngrok.set_auth_token("1au2Q4xVgIFOFTQyuKxsRPNDbMs_2yvVEyV96eeYXWFh2sMvk")

# public_url = ngrok.connect(5173)
# print("Public URL:", public_url.public_url)


from pyngrok import ngrok

# Close any existing tunnels managed by this Python process
ngrok.kill()

try:
    tunnel = ngrok.connect(5173, "http")
    print("Public URL:", tunnel.public_url, flush=True)
    print("Tunnel is running. Press Ctrl+C to stop.", flush=True)

    # Keep the tunnel alive
    ngrok.get_ngrok_process().proc.wait()

except KeyboardInterrupt:
    print("\nStopping ngrok...")

finally:
    ngrok.kill()
