from main import classifier2, decoder
from prepare_data import load_one_data
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import sys
import os
import time

keyword = ["down", "go", "left", "no", "off","on", "right", "stop", "up", "yes", "filler"]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    C2 = classifier2()
    C2.to(device)
    C2.load_state_dict(torch.load('encoder.pkl'))
    D = decoder()
    D.to(device)
    D.load_state_dict(torch.load('decoder.pkl'))
    
    C2.eval()
    D.eval()
    
    startTime = time.time()
    for i in range(1):
        data = torch.tensor(load_one_data(sys.argv[1]))
        data = data.unsqueeze(0).to(device).float()
        output = D(C2(data))
        predict = torch.max(output, 1)[1]
    endTime = time.time()
    print("{}, time: {:.3f}s".format(keyword[predict], endTime - startTime))

if __name__ == "__main__": main()