'''
# Step55 CNN 메커니즘(1)
'''
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero.utils import get_conv_outsize

if __name__ == '__main__':
    H, W = 4, 4
    KH, KW = 3, 3
    SH, SW = 1, 1
    PH, PW = 1, 1

    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    print(OH, OW)
