安裝套件:
```
conda:
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip:
pip install -r requirement.txt
```

測試自己的音檔or全部音檔:
```
test_one_data:
python test_one_data.py <audio path>
test:
python testing.py
```

重新訓練:
```
特徵抽取:
python prepare_data.py <dataset path> <output feature path>
訓練:
python main.py <feature path>
```