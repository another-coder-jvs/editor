from google.colab import drive
drive.mount('/content/drive')


!sudo apt-get update -qq
!sudo apt-get install -y zstd pv

!pip install pyngrok
!sudo apt-get update -y
!sudo apt-get install python3.11 python3.11-venv python3.11-dev -y

!pv /content/drive/MyDrive/project_folders/venv.tar.zst | tar -I zstd -xf - -C /content/


