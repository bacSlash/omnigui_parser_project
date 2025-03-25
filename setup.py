from setuptools import setup, find_packages


setup(
    name='omnigui-parser',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'easyocr',
        'torchvision',
        'supervision==0.18.0',
        'transformers',
        'ultralytics==8.3.70',
        'numpy==1.26.4',
        'openai',
        'opencv-python',
        'opencv-python-headless',
        'gradio',
        'dill',
        'accelerate',
        'timm',
        'einops==0.8.0',
        'paddlepaddle',
        'paddleocr',
        'pathlib',
    ],
    include_packages_data=True,
    package_data={
        'omnigui_parser': [
            'weights/icon_detect/*',
            'utils.py*',
            'util/*',
        ],
    },
    entry_points={
        'console_scripts': [
            'video_extractor=omnigui_parser.video_extractor:main',
            'multi_img_parser=omnigui_parser.multi_img_parser:main',
        ],
    },
)
