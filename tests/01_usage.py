import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) +
                "/..")
sys.path.append('..')

import numpy
import texturesynth as tex
import cv2
import pyrtools as pyr


filename = sys.argv[1]
scales = int(sys.argv[2])
ori = int(sys.argv[3])
neighbor = int(sys.argv[4])
img = cv2.imread(filename, 0)
pss = tex.analyzer.analyze(img, scales, ori, neighbor)

synthesizer = tex.synthesizer.Synthesizer(pss, (128, 128))
synthesizer.next()
