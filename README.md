# yolo-nas-object-blurring

### Steps to run Code
- Clone the repository.
```
git clone git@github.com:mmoollllee/yolo-nas-object-blurring.git
```
- Goto the cloned folder.
```
cd yolo-nas-object-blurring
```
- Install requirements
```
pip install -r requirements.txt
```
- Create a config.txt to set default parameters
```
[Main]
source = webcam
dest = processed
conf_thres = 0.1
blurratio = 10
```
- Run the code
```
python detect_and_blur.py
```
- Output file will be created in the <b>dest</b> with original filename.
