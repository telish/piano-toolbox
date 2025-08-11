import cv2
import numpy as np
import glob
import json

# Anpassen: Anzahl der inneren Ecken pro Zeile/Spalte (z. B. 7x6)
CHECKERBOARD = (10, 7)

# 3D-Punkte: Schachbrett liegt in der XY-Ebene (Z = 0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Listen für Kalibrierung
objpoints = []  # 3D Punkte in realer Welt
imgpoints = []  # 2D Punkte im Bild

# Lade alle Bilder im Ordner
images = glob.glob("calibration/checkerboard/*.png")

gray_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow("Chessboard", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

if gray_shape is None:
    print("No images in calibration/checkerboard")
    exit(-1)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

# Ergebnisse speichern
camera_params = {
    "camera_matrix": mtx.tolist(),
    "distortion_coefficients": dist.tolist(),
}

with open("calibration/checkerboard/camera_params.json", "w") as f:
    json.dump(camera_params, f, indent=4)

