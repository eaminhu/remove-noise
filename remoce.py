'''
Description: 
Author: HM
Email: ming@cpglobalinnovation.com
Date: 2024-06-06 18:58:28
LastEditTime: 2024-06-06 19:00:06
'''
import cv2
import numpy as np

# Initialize global variables
g_bDrawing = False
g_CurrPoint = (-1, -1)
g_OrgPoint = (-1, -1)
g_nThick = 3
g_nBlue = 255
g_nGreen = 255
g_nRed = 0

# Mouse callback function
def onMouse(event, x, y, flags, param):
    global g_bDrawing, g_CurrPoint, g_OrgPoint, srcImage, maskImage

    if event == cv2.EVENT_MOUSEMOVE:
        g_OrgPoint = g_CurrPoint
        g_CurrPoint = (x, y)
        if g_bDrawing:
            cv2.line(srcImage, g_OrgPoint, g_CurrPoint, (g_nBlue, g_nGreen, g_nRed), g_nThick)
            cv2.imshow("Original Image - Draw Spots with Mouse", srcImage)
            cv2.line(maskImage, g_OrgPoint, g_CurrPoint, (255, 255, 255), g_nThick)  # Use white for the mask
            cv2.imshow("Mask Image", maskImage)

    elif event == cv2.EVENT_LBUTTONDOWN:
        g_bDrawing = True
        g_OrgPoint = (x, y)
        g_CurrPoint = g_OrgPoint

    elif event == cv2.EVENT_LBUTTONUP:
        g_bDrawing = False

def main():
    global srcImage, maskImage

    srcImage = cv2.imread("C:/Users/86186/Desktop/removeNoise/input.jpg")
    g_srcImage = srcImage.copy()

    # Create a mask image
    maskImage = np.zeros(srcImage.shape[:2], dtype=np.uint8)

    cv2.namedWindow("Original Image - Draw Spots with Mouse")
    cv2.setMouseCallback("Original Image - Draw Spots with Mouse", onMouse)

    while True:
        cv2.imshow("Original Image - Draw Spots with Mouse", srcImage)
        key = cv2.waitKey(0)

        if key == 27:  # Press 'Esc' to exit the program
            break

        if key == ord('1'):
            srcImage = g_srcImage.copy()
            maskImage = np.zeros(srcImage.shape[:2], dtype=np.uint8)
            cv2.imshow("Original Image - Draw Spots with Mouse", srcImage)

        if key == 13:  # Press 'Enter' to start inpainting
            dstImage = cv2.inpaint(srcImage, maskImage, 3, cv2.INPAINT_TELEA)
            cv2.imshow("Repaired Image", dstImage)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
