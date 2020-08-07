from main import classifier2, decoder, AudioDataset
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import sys
import os

keyword = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "filler"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(input_path):
    test_x = np.load(os.path.join(input_path, "test_X.npy"))
    test_y_spk = np.load(os.path.join(input_path, "test_Y_spk.npy"))
    test_y_text = np.load(os.path.join(input_path, "test_Y_text.npy"))
    return test_x, test_y_spk, test_y_text

def test(test_loader, encoder, decoder):
    test_acc = []
    for _, (mfccs, target_spk, target_text) in enumerate(test_loader):
        mfccs_cuda = mfccs.to(device).float()
        target_cuda_text = target_text.to(device)

        output = decoder(encoder(mfccs_cuda))

        predict = torch.max(output, 1)[1]
        acc = np.mean((target_cuda_text == predict).cpu().numpy())

        test_acc.append(acc)
    
    print(1 - np.mean(test_acc))

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    test_x, test_y_spk, test_y_text = load_data(sys.argv[1])

    test_dataset = AudioDataset(test_x, test_y_spk, test_y_text)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 32, shuffle = False)

    C2 = classifier2()
    C2.to(device)
    C2.load_state_dict(torch.load('encoder.pkl'))
    D = decoder()
    D.to(device)
    D.load_state_dict(torch.load('decoder.pkl'))
    
    C2.eval()
    D.eval()

    test(test_loader, C2, D)

    

if __name__ == "__main__": main()