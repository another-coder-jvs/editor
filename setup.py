from setuptools import setup, find_packages

setup(
    name="ai-image-editor-backend",
    version="1.0.0",
    packages=find_packages(where="backend"),
    package_dir={"": "backend"},
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.30.1",
        "python-multipart>=0.0.9",
        "pydantic>=2.7.4",
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "transformers>=4.42.0",
        "diffusers>=0.29.0",
        "accelerate>=0.31.0",
        "opencv-python-headless>=4.10.0",
        "Pillow>=10.3.0",
        "numpy>=1.26.4",
        "rembg[gpu]>=2.0.57",
        "basicsr>=1.4.2",
        "realesrgan>=0.3.0",
    ],
    extras_require={
        "dev": ["pytest>=8.2.2", "pytest-asyncio>=0.23.7"],
    },
)
