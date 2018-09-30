import pytesseract
import cv2
import numpy as np
import os
from PIL import Image, ImageSequence
import time
import itertools
import sys
import imutils
import multiprocessing


""" verCredit = ["Aboño", "credit note", "actura rectificativa"]
verNo = ["Numero de Factura", "Number", "Numero Factura",
         "Factura", "Nº factura", "Invoice Number"]
verDate = ["Fecha de Factura", "Fecha Factura",
           "Fecha", "Invoice date", "Date","TransactionDate"]
verTotal = ["Total", "Total Valor", "SUB Total",
            "Total a pagar", "Total con", "Total Factura"]
verCIF = ["NIF", "CIF", "Customer VAT Number", "VAT Number", "NIFCIF","VAT#"]
verRate = ["Conversion", "tipo de cambio"]
verOrderNo = ["Pedido", "PO Number", "Nº DE PEDIDO",
              "P.O.", "Nº", "Order No", "Sales Order No,PO"]
verTax = ["Tax","IVA","Impuesto","%Impuesto"] """
verNo = ["Number"]
verCredit = ["Credit"]
verDate = ["Date"]
verTotal = ["Total"]
verCIF = ["VAT"]
verRate = ["Conversion"]
verOrderNo = ["PO"]
verTax = ["Tax"]
ver = verCredit + verNo + verDate + verTotal + \
    verCIF + verRate + verOrderNo + verTax
for text in ver:
    text = ''.join(c for c in text if c not in ",.'`-}_*;=|+!‘>—:~°][{")


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def removeTables(img): #removes tables from image to improve ocr accuracy.
    horizontal_img = img.copy()
    vertical_img = img.copy()
    _, horizontal_img = cv2.threshold(
        horizontal_img, 0, 255, cv2.THRESH_BINARY)
    _, vertical_img = cv2.threshold(vertical_img, 0, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal_img = cv2.erode(horizontal_img, kernel, iterations=1)
    horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_img = cv2.erode(vertical_img, kernel, iterations=1)
    vertical_img = cv2.dilate(vertical_img, kernel, iterations=1)
    mask_img = horizontal_img + vertical_img
    _, mask_img = cv2.threshold(
        mask_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask_img = cv2.dilate(mask_img, se, iterations=1)
    return mask_img


def returnRotated(img): #checks if image orientation is correct and returns desired orientation.
    cv2.imwrite("temp.tif", img)
    x = os.system('tesseract temp.tif temp -l osd --psm 0')
    os.remove("temp.tif")
    f = open('temp.osd', 'r')
    x = f.read()
    x1 = int(x.split(':')[2].split('\n')[0])
    print(x1)
    if(x1 == 90):
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif(x1 == 270):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif(x1 == 180):
        img = cv2.rotate(img, cv2.ROTATE_180)
    f.close()
    os.remove("temp.osd")
    return img


def tessDraw(tc, fc, cnt, src):

    try:
        global ver
        spl = set(",.'`-}_*;=|+/!‘>—:~°][{»¢")
        x, y, w, h = cv2.boundingRect(cnt)
        centre = (int(x+(w/2)), int(y + (h/2)))
        if(w > 5 and h > 5):
            im = cv2.getRectSubPix(src, (w, h), centre)
            im = cv2.copyMakeBorder(im, top=10, bottom=10, left=10, right=10,
                                    borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
           
            data = pytesseract.image_to_data(Image.fromarray(
                im), config="--psm 7 --oem 1", output_type="dict")
            for i, item in enumerate(data['text']):
                data['text'][i] = data['text'][i].strip()
                if(set(data['text'][i]).issubset(spl)):
                    pass
                elif(data['text'][i] != "" and data['height'][i] > 5 and data['width'][i] > 5 and data['conf'][i] > 50):
                    x1 = data['left'][i]
                    y1 = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    conf = data['conf'][i]
                    text = data['text'][i].strip()
                    t1 = ''.join(
                        c for c in text if c not in "»,.'`-}:/_*;=|+!‘>!/—:~°][{¢")
                    flag1 = True
                    flag2 = True
                    key = t1 + "-" + str(x+x1-10) + "-"+str(y+y1-10)
                    i = [range(-5, 5, 1)]
                    comb = list(itertools.product(*i, repeat=2))
                    for obj in comb:
                        key1 = t1 + "-" + \
                            str(x+x1-10+obj[0]) + "-"+str(y+y1-10+obj[1])
                        if(key1 in tc.keys() or key1 in fc.keys()):
                            flag1 = False
                            break
                    if(flag1 == True):
                        for i in range(len(ver)):
                            obj = ver[i]
                            if(obj.lower().startswith(t1.lower()) and len(t1.lower()) > len(obj)/2):
                                fc[key] = [(x+x1-10), (y+y1-10),
                                       w+1, h+1, text, conf]
                                flag2 = False
                                break
                        if(flag2 == True):
                            tc[key] = [(x+x1-10), (y+y1-10),
                                w+1, h+1, text, conf]
            return True
        else:
            return -2
    except Exception as e:
        raise (Exception)
        # sys.exit()


def findC(tc, fc, finalcnts, img, src):
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 2))
    img = cv2.dilate(img, se, iterations=1)
    _, contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pool = multiprocessing.Pool()
    for cnt in contours:
        try:
            res = pool.apply_async(tessDraw, args=(tc, fc, cnt, src,))
        except Exception as e:
            print("here")
            pool.terminate()
            pool.close()
            print("broke out of loop")
            sys.exit()
        try:
            res.get(timeout=5)
        except TimeoutError as e1:
            print("y")
            pool.terminate()
            sys.exit()
        if(type(res.get()) != int):
            if(res.get() == True):
                finalcnts.append(cnt)
                # tc.append(item)
                """if(tc):
                        for e in tc:
                            if(e[0]>=(item[0]-5) and e[0]<=(item[0]+5) and e[1] >= (item[1]-5) and e[1] <= (item[1]+5)):
                                pass
                            else:
                                tc.append(item)
                    else: """
            else:
                pool.terminate()
                pool.close()
                print('exiting')
                sys.exit()
    print("done")
    pool.terminate()
    pool.close()

def FindKeys(fc,vc,test,configdict):
    global verCredit,verNo,verDate,verTotal,verCIF, verRate,verOrderNo,verTax
    for key, textval in fc.items():
            key = ''.join(c for c in textval[4] if c not in "»,.'`-}:/_*;=|+!‘>!/—:~°][{¢")
            if(any(obj.lower().startswith(key.lower()) for obj in verNo)):
                for key1,numval in vc.items():
                    if(len(numval[4])==configdict['Invoice No'][4] and numval[0] > textval[0] + configdict['Invoice No'][0] and numval[0] < textval[0] + configdict['Invoice No'][1] and numval[1] > textval[1] + configdict['Invoice No'][2] and numval[1] < textval[1] + configdict['Invoice No'][3] and numval[4] not in test['Invoice No']):
                        test['Invoice No'].append(numval[4])
            elif(any(obj.lower().startswith(key.lower()) for obj in verDate)):
                for key1,numval in vc.items():
                    if(numval[0] > textval[0] + configdict['Date'][0] and numval[0] < textval[0] + configdict['Date'][1] and numval[1] > textval[1] + configdict['Date'][2] and numval[1] < textval[1] + configdict['Date'][3] and numval[4] not in test['Date']):
                        test['Date'].append(numval[4])
            elif(any(obj.lower().startswith(key.lower()) for obj in verTotal)):
                for key1,numval in vc.items():
                    if(numval[0] > textval[0] + configdict['Total'][0]  and numval[0] < textval[0] + configdict['Total'][1] and numval[1] > textval[1] + configdict['Total'][2] and numval[1] < textval[1] + configdict['Total'][3] and numval[4] not in test['Total']):
                        if("$" in numval[4] and "USD" not in test['Currency']):
                            test['Currency'].append("USD")
                        elif("€" in numval[4]and "EUR" not in test['Currency']):
                            test['Currency'].append("EUR")
                        elif("USD" in numval[4]and "USD" not in test['Currency']):
                                test['Currency'].append("USD")
                        elif("EUR" in numval[4]and "EUR" not in test['Currency']):
                                test['Currency'].append("EUR")
                        else:
                            test['Total'].append(numval[4])
            elif(any(obj.lower().startswith(key.lower()) for obj in verCIF)):
                for key1,numval in vc.items():
                    if(numval[0] > textval[0] + configdict['CIF'][0]  and numval[0] < textval[0] + configdict['CIF'][1] and numval[1] > textval[1] + configdict['CIF'][2] and numval[1] < textval[1] + configdict['CIF'][3] and numval[4] not in test['CIF']):
                        test['CIF'].append(numval[4])
            elif(any(obj.lower().startswith(key.lower()) for obj in verRate)):
                for key1,numval in vc.items():
                    if(numval[0] > textval[0] + configdict['Conversion Rate'][0]  and numval[0] < textval[0] + configdict['Conversion Rate'][1] and numval[1] > textval[1] + configdict['Conversion Rate'][2] and numval[1] < textval[1] + configdict['Conversion Rate'][3] and numval[4] not in test['Conversion Rate']):
                        test['Conversion Rate'].append(numval[4])
            elif(any(obj.lower().startswith(key.lower()) for obj in verOrderNo)):
                for key1,numval in vc.items():
                    if(numval[0] > textval[0] + configdict['Order No'][0] and numval[0] < textval[0] + configdict['Order No'][1] and numval[1] > textval[1] + configdict['Order No'][2] and numval[1] < textval[1]+configdict['Order No'][3] and numval[4] not in test['Order No']):
                        test['Order No'].append(numval[4])
            elif(any(obj.lower().startswith(key.lower()) for obj in verTax)):
                for key1,numval in vc.items():
                    if(numval[0] > textval[0] + configdict['Tax'][0] and numval[0] < textval[0] + configdict['Tax'][1] and numval[1] > textval[1] + configdict['Tax'][2] and numval[1] < textval[1]+ configdict['Tax'][3] and numval[4] not in test['Tax']):
                        test['Tax'].append(numval[4])
            elif(any(obj.lower().startswith(key.lower()) for obj in verCredit)):
                for key1,numval in vc.items():
                    if(numval[0] > textval[0] + configdict['Credit'][0] and numval[0] < textval[0] + configdict['Credit'][1] and numval[1] > textval[1] +configdict['Credit'][2] and numval[1] < textval[1]+ configdict['Credit'][3] and numval[4] not in test['Credit']):
                        test['Credit'].append(numval[4])
    return test

def process(img):
    img = img.copy()
    img = returnRotated(img)
    img = imutils.resize(img, img.shape[1]-1)
    res = img.copy()
    res1 = img.copy()
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(
        res, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_img = removeTables(res)
    output = cv2.connectedComponentsWithStats(mask_img, 8, cv2.CV_8U)
    num_stats = output[0]
    labels = output[1]
    stats = output[2]
    for label in range(num_stats):
        if((int(stats[label][4]/3) < 500000)):
            res[labels == label] = 0
    res = cv2.bitwise_not(res)
    v = np.median(img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # DENOISING VALUE NEEDS TO BE ADJUSTED TO GET RESULTS
    denoise = cv2.medianBlur(img, 5)
    dn = denoise.copy()
    dn = adjust_gamma(dn, 3)
    dn = cv2.cvtColor(dn, cv2.COLOR_BGR2GRAY)
    _, dn = cv2.threshold(
        dn, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_img = removeTables(dn)
    output = cv2.connectedComponentsWithStats(mask_img, 8, cv2.CV_8U)
    num_stats = output[0]
    labels = output[1]
    stats = output[2]
    for label in range(num_stats):
        if((int(stats[label][4]/3) < 500000)):
            dn[labels == label] = 0
    dn = cv2.bitwise_not(dn)
    enhance = adjust_gamma(img, 0.001)
    eh = enhance.copy()
    eh = cv2.cvtColor(eh, cv2.COLOR_BGR2GRAY)
    _, eh = cv2.threshold(
        eh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_img = removeTables(eh)
    output = cv2.connectedComponentsWithStats(mask_img, 8, cv2.CV_8U)
    num_stats = output[0]
    labels = output[1]
    stats = output[2]
    for label in range(num_stats):
        if((int(stats[label][4]/3) < 500000)):
            eh[labels == label] = 0
    eh = cv2.bitwise_not(eh)
    # img
    img = cv2.Canny(img, lower, upper, L2gradient=True)
    mask_img = removeTables(img)
    output = cv2.connectedComponentsWithStats(mask_img, 8, cv2.CV_8U)
    num_stats = output[0]
    labels = output[1]
    stats = output[2]
    for label in range(num_stats):
        if((int(stats[label][4]/3) < 500000)):
            img[labels == label] = 0
    # denoise
    denoise = cv2.Canny(denoise, lower, upper, L2gradient=True)
    mask_img = removeTables(denoise)
    output = cv2.connectedComponentsWithStats(mask_img, 8, cv2.CV_8U)
    num_stats = output[0]
    labels = output[1]
    stats = output[2]
    for label in range(num_stats):
        if((int(stats[label][4]/3) < 500000)):
            denoise[labels == label] = 0
    # enhance
    enhance = cv2.Canny(enhance, lower, upper, L2gradient=True)
    mask_img = removeTables(enhance)
    output = cv2.connectedComponentsWithStats(mask_img, 8, cv2.CV_8U)
    num_stats = output[0]
    labels = output[1]
    stats = output[2]
    for label in range(num_stats):
        if((int(stats[label][4]/3) < 500000)):
            enhance[labels == label] = 0

    print('start of mp')
    cnts = [res, dn, eh]
    imgs = [img, denoise, enhance]
    vc = {}
    allow = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$/¢€-")
    with multiprocessing.Manager() as manager:
        finalcnts = manager.list()
        tc = manager.dict()
        fc = manager.dict()
        temp = []
        for img, cnt in zip(imgs, cnts):
            try:
                p = multiprocessing.Process(
                    target=findC, args=(tc, fc, finalcnts, img, cnt,))
                p.daemon = False
                temp.append(p)
                p.start()
            except Exception as e:
                raise Exception
                print("broke out of loop")
                sys.exit()
        for proc in temp:
            proc.join()
            proc.terminate()
        for key, value in fc.items():
            cv2.rectangle(
                res1, (value[0], value[1]), (value[0]+value[2], value[1]+value[3]), (0, 0, 255), 2)
        for key, value in tc.items():
            if(set(value[4]).issubset("0123456789,.-$/¢€") or (value[4].isalpha() == False and value[4].isalnum()) or (value[4].isalnum() == False and set(value[4]).issubset(allow)) or value[4]== "USD" or value[4]=="EUR"):
                vc[key] = value
                cv2.rectangle
                (res1, (value[0], value[1]), (value[0]+value[2], value[1]+value[3]), (255, 0, 0), 2)
            else:
                cv2.rectangle(
                    res1, (value[0], value[1]), (value[0]+value[2], value[1]+value[3]), (0, 255, 0), 1)
        x1c = dict(fc)
        x2c = dict(vc)
        return res1,x1c,x2c


if __name__ == '__main__':
    start_time = time.clock()
    valuedict = {'Total': [0,900,-10,10], 'Invoice No': [-50,20,0,75,10], 'Credit': [-10,10,-10,10],
        'Date': [-150,0,0,70], 'CIF': [0,130,-10,10], 'Conversion Rate': [0,815,-10,10], 'Tax': [-80,100,0,120], 'Order No': [-50,50,0,75]}
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    image_path = ""
    test = {'Total': [], 'Invoice No': [], 'Credit': [], 'Currency': [],
        'Date': [], 'CIF': [], 'Conversion Rate': [], 'Tax': [], 'Order No': []}
    img = Image.open(image_path)
    for i, page in enumerate(ImageSequence.Iterator(img)):
        page = page.convert("RGB")
        p = np.asarray(page)
        pageno = i
        write = "test%s.tif" % pageno
        print(write)
        res,fc,vc = process(p)
        print(time.clock() - start_time)
        cv2.imwrite(write, res)
        chk = time.clock()
        test = FindKeys(fc,vc,test,valuedict)
        print(test)
        print("%s s"%(time.clock()-chk))
    print(test)
#numbers = set("1234567890,.()$")
#print(time.clock() - start_time)
# print("s")
