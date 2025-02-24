from setuptools import setup, find_packages

setup(
    name="AS_GeminiCaptioning",
    version="1.0.0",
    description="Node for ComfyUI that builds a multimodal prompt from text inputs and an image, sends it to Google Gemini via the google-generativeai SDK, and returns the textual response.",
    author_email="art.svetozarov@gmail.com",
    url="https://github.com/svetozarov/AS_GeminiCaptioning", 
    packages=find_packages(),
    install_requires=[
        "Pillow",
        "requests",
        "google-generativeai"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
