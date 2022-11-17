from PIL import Image
import numpy as np
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
filename = 'Test Images\PUNJABI\Punjabi1.jpg'
img1 = np.array(Image.open(filename))
text = pytesseract.image_to_string(img1,lang='pan')
print(text)

# img='English.jpg'
# osd = pytesseract.image_to_osd(img)
# script = re.search("Script: ([a-zA-Z]+)\n", osd).group(1)
# if(script=="Devanagari"):
#   text = pytesseract.image_to_string(img,lang='hin')
# elif(script=="Latin"):
#   text = pytesseract.image_to_string(img,lang='eng')



