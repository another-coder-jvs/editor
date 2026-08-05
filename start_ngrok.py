

from pyngrok import ngrok
ngrok.set_auth_token("3HRjpFMqfHffIlX0oPqHITBQalK_457XLfBzcPqizqijAyV1p")

ngrok.kill()
tunnel2 = ngrok.connect(
    8000,
    proto="http"
)

print(tunnel2.public_url)
print(tunnel2.public_url+"/docs")

print("Need drive permission to store the masks !" )
from google.colab import drive
drive.mount('/content/drive')
