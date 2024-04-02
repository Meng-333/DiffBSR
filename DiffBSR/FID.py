import argparse
import os
from litsr.metrics import *

#scene = 'tenniscourt'

rslt_path = "/root/autodl-tmp/FastDiffBSR/FID/SR"
#hr_path =  "/root/autodl-tmp/FastDiffBSR/load/AIRS_test"
hr_path = "/root/autodl-tmp/FastDiffBSR/load/WHU_test"

#hr_path =  "/root/autodl-tmp/FastDiffBSR/load/UCM_test/{}".format(scene)

paths = [rslt_path, hr_path]
fid_score = calc_fid(paths)
print("- SR_FID : {:.5f}".format(fid_score))
print16 = "- SR_FID : {:.5f}".format(fid_score)



