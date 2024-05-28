# install_dependencies.py

import os

# Install tesseract and pytesseract
os.system('apt-get install tesseract-ocr')
os.system('pip install pytesseract')

# Install ArabicOcr and its dependencies
os.system('pip install ArabicOcr')

# Clone the card-rectification repository
