echo "Installing Dependancies"
pip install -r requirements.txt


echo "Installing Ngrok"
pip -q install fastapi uvicorn pyngrok nest_asyncio
cd backend

echo "Now First start NGROK with start_ngrok.py ! \nthen Backend with python main.py"

echo "Need drive permission to store the masks !" 
from google.colab import drive
drive.mount('/content/drive')