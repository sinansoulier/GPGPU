# nix-shell shell.nix
rm -rf build/ *.mp4* *.so*                       # 2
cmake -S . -B build --preset release -D USE_CUDA=ON  # 2 (ou debug)
cmake --build build                                  # 2


# wget https://gstreamer.freedesktop.org/media/sintel_trailer-480p.webm # 3
export GST_PLUGIN_PATH=$(pwd)                                         # 4
# ln -s ./build/libgstcudafilter-cpp.so libgstcudafilter.so             # 5
ln -s ./build/libgstcudafilter-cpp.so libgstcudafilter.so            # 5

# gst-launch-1.0 uridecodebin uri=file://$(pwd)/sintel_trailer-480p.webm ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=video.mp4 #5
gst-launch-1.0 uridecodebin uri=file://$(pwd)/subject/camera.mp4 ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=video.mp4 #5