import cv2

img_file = ""
img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE) # load in grayscale
edges = cv2.Canny(img,100,200)

cv2.imshow('Edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()




