pip install opencv-python opencv-contrib-python


💡 How it Works

    Starts webcam → detects faces using Haar Cascades.

    Press d → face is saved with a name.

    Model is re-trained instantly.

    When shown again, the app uses LBPH face recognition to match and show your name.

🔁 To Reset or Re-train

Delete:

    All .jpg files in faces/

    trainer/ folder contents (trainer.yml, labels.txt)

Then run again.


logs...


emcoda@crocodile:~/Desktop/Documents/Fcedetect$ python main.py
[ WARN:0@0.051] global cap_v4l.cpp:914 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
[ERROR:0@0.051] global obsensor_uvc_stream_channel.cpp:163 getStreamChannelGroup Camera index out of range
emcoda@crocodile:~/Desktop/Documents/Fcedetect$ python main.py
Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. Use QT_QPA_PLATFORM=wayland to run on Wayland anyway.
Enter name to save face as: england
[INFO] Saved face as faces/england.jpg
[INFO] Training complete.
[1]+  Killed                  python main.py
emcoda@crocodile:~/Desktop/Documents/Fcedetect$ 

