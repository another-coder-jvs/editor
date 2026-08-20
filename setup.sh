#!/usr/bin/env bash
# Copy and extract venv only if it hasn't already been extracted
if [ ! -d "./venv" ]; then
    echo -e "\nCopying Zip......"
    cp /content/drive/MyDrive/project_folders/venv.zip .

    echo -e "\nExtracting venv......"
    unzip -q venv.zip

    echo -e "\nCompleted......"
else
    echo -e "\nvenv already exists. Skipping copy/extract......"
fi


# Install pyngrok only if it isn't already installed
if ! python -c "import pyngrok" 2>/dev/null; then
    echo -e "\nInstalling pyngrok......"
    pip install pyngrok
    echo -e "\nInstalling pyngrok completed......"
else
    echo -e "\npyngrok already installed. Skipping......"
fi


# Install Python 3.11 and related packages only if they're missing
if command -v python3.11 >/dev/null 2>&1 && \
   dpkg -s python3.11-venv >/dev/null 2>&1 && \
   dpkg -s python3.11-dev >/dev/null 2>&1; then

    echo -e "\nPython 3.11, python3.11-venv and python3.11-dev already installed. Skipping......"

else
    echo -e "\nUpdating packages and installing Python 3.11 with python3.11-venv and python3.11-dev......"

    sudo apt-get update -y
    sudo apt-get install python3.11 python3.11-venv python3.11-dev -y

    echo -e "\nPython installation completed......"
fi

# echo "\nCopying Zip...... "
# cp /content/drive/MyDrive/project_folders/venv.zip . && unzip venv.zip

# echo "\nCompleted...... "

# echo "\nInstalling pyngrok...... "
# pip install pyngrok
# echo "\nInstalling pyngrok completed...... "
# echo "\nUpdating packages and installing python3.11 with python3.11-venv ,  python3.11-dev...... "
# sudo apt-get update -y
# sudo apt-get install python3.11 python3.11-venv python3.11-dev -y









#  ======================= Old...
# from setuptools import setup, find_packages

# setup(
#     name="ai-image-editor-backend",
#     version="1.0.0",
#     packages=find_packages(where="backend"),
#     package_dir={"": "backend"},
#     python_requires=">=3.11",
#     install_requires=[
#         "fastapi>=0.111.0",
#         "uvicorn[standard]>=0.30.1",
#         "python-multipart>=0.0.9",
#         "pydantic>=2.7.4",
#         "torch>=2.3.0",
#         "torchvision>=0.18.0",
#         "transformers>=4.42.0",
#         "diffusers>=0.29.0",
#         "accelerate>=0.31.0",
#         "opencv-python-headless>=4.10.0",
#         "Pillow>=10.3.0",
#         "numpy>=1.26.4",
#         "rembg[gpu]>=2.0.57",
#         "basicsr>=1.4.2",
#         "realesrgan>=0.3.0",
#     ],
#     extras_require={
#         "dev": ["pytest>=8.2.2", "pytest-asyncio>=0.23.7"],
#     },
# )
