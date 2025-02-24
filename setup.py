from setuptools import setup, find_packages

setup(
    name="AS_GeminiCaptioning",
    version="1.0.2",
    description="A ComfyUI node that combines an image with simple text parameters to create a prompt, sends it to the Google Gemini API via the google-generativeai SDK, and returns the generated text response along with the original prompt and an execution log",
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
