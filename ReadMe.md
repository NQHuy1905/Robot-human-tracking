# Vision task

A vision human tracking for edge robot

## Usage

Save check point in ./checkpoint

For new id to search 
Collect new data
```bash
python get_newdata.py
```
Update new data to Milvus server
```bash
python ./utils/milvus_tool.py
```
Running script

```bash
python python track_stream_v3.py --yolo-weights ./checkpoint/person_detection/yolov8n-seg.engine --tracking-method bytetrack --device 0 --classes 0  --id-to-send hoang --streaming-host 224.0.0.1 --multicast True --laser-power-set 360
```

To watch stream on other machine, run
```bash
gst-launch-1.0 udpsrc auto-multicast=true address=224.0.0.1 port=5000 multicast-iface=wlp1s0 ! "application/x-rtp, media=(string)video, encoding-name=(string)H265" ! rtph265depay ! h265parse ! avdec_h265 ! glimagesink sync=true async=true -e -v
```

Demo

https://github.com/user-attachments/assets/b7cbc3a1-ae3f-4847-b5c9-f7bd4d4b8832


