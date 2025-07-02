from setuptools import find_packages, setup

setup(
    name="deepx-open-modelzoo",
    version="0.1.0",
    author="DeepX",
    author_email="jwjung@deepx.ai, dhe@deepx.ai",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["*"]),
    python_requires=">=3.11",
    entry_points={"console_scripts": ["dxmz=dx_modelzoo.main:main"]},
    install_requires=[
        "torch==2.3.1",
        "torchvision==0.18.1",
        "onnx==1.17.0",
        "onnxruntime==1.20.0",
        "opencv-python>=4.10.0.84",
        "tqdm>=4.67.1",
        "pycocotools>=2.0.8",
        "scipy==1.15.1",
        "numpy==1.26.4",
        "loguru",
        "requests",
        "prefect==3.3.1",
        "boto3"
    ],
)
