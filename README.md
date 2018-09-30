# zonal-document-ocr
Project to identify locations of certain parameters in a tif format scanned document and retrieve their values.
Parameter values (verNo, verDate, etc) must be changed to suit format of document.
valuedict gives pixel range to search in to find values that match a certain key.
Requires tesseract OCR, pytesseract wrapper, opencv-python, numpy, PIL.
