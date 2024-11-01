import numpy as np
import cv2

def read_P2_from_calib(calib_file):
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('P2:'):
                # split the line and convert strings to float
                values = [float(x) for x in line.strip().split()[1:]]
                # reshape the list into a 3x4 matrix
                P2 = np.array(values).reshape(3, 4)
                return P2
    raise ValueError(f"P2 matrix not found in {calib_file}")

P2 = read_P2_from_calib('CSCI739/samples/calibration/000044.txt')
print(P2)

k1, r1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)
t1 = t1/t1[3]

#displaying the results
print('Intrinsic:\n', k1)
print('Rotation:\n', r1)
print('Translation:\n', t1.round(4))

