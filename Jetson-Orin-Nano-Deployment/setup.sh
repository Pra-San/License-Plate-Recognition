sudo apt update

sudo apt upgrade

sudo apt-cache show nvidia-jetpack

nvcc --version

sudo apt-get install -y libopenblas-base libopenmpi-dev


wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O torch-1.11.0-cp38-cp38-linux_aarch64.whl

pip3 install torch-1.11.0-cp38-cp38-linux_aarch64.whl

sudo apt install -y libjpeg-dev zlib1g-dev

git clone --branch v0.13.0 https://github.com/pytorch/vision torchvision

cd torchvision

python3 setup.py install --user

cd ..

git clone https://github.com/WongKinYiu/yolov7.git

pip3 install -r requirements_modified.txt

pip3 install easyocr

echo "Downloading Flask for Testing"
pip3 install flask

cp detect-plates.py ./yolov7/
cp detect_plates_single_gate.py ./yolov7/
cd ./yolov7/utils
rm datasets.py
cd ../..
cp datasets.py ./yolov7/utils/

cd ./yolov7

git clone https://github.com/abewley/sort.git






