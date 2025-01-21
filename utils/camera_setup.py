import cv2
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib
import pyrealsense2 as rs


class GstreamerPython:
    def __init__(self, streaming_host):
        self.streaming_host = streaming_host
    # GStreamer setup
    def setup_gstreamer(self):
        Gst.init(None)
        single_pipeline = Gst.Pipeline.new("webcam-stream")
        appsrc = Gst.ElementFactory.make("appsrc", "source")
        nvvideoconvert = Gst.ElementFactory.make("nvvidconv", "nvvideoconvert")
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        parser = Gst.ElementFactory.make("h265parse", "parser")
        payloader = Gst.ElementFactory.make("rtph265pay", "payloader")
        sink = Gst.ElementFactory.make("udpsink", "sink")

        if not all([single_pipeline, appsrc, nvvideoconvert,
                    encoder, parser, payloader, sink]):
            print("Not all elements could be created.")
            return None

        

        single_pipeline.add(appsrc)
        single_pipeline.add(nvvideoconvert)
        single_pipeline.add(encoder)
        single_pipeline.add(parser)
        single_pipeline.add(payloader)
        single_pipeline.add(sink)

        if not (appsrc.link(nvvideoconvert) and
                nvvideoconvert.link(encoder) and encoder.link(parser) and
                parser.link(payloader) and payloader.link(sink)):
            print("Elements could not be linked.")
            return None

        appsrc.set_property("caps", Gst.Caps.from_string("video/x-raw,format=I420,width=1280,height=720,framerate=30/1"))
        encoder.set_property("bitrate", 10000)
        payloader.set_property("config-interval", 1)
        sink.set_property("host", self)
        sink.set_property("port", 5000)
        return single_pipeline, appsrc

    def setup_gstreamer_multicast(self):
        Gst.init(None)
        multicast_pipeline = Gst.Pipeline.new("webcam-stream")
        appsrc = Gst.ElementFactory.make("appsrc", "source")
        nvvideoconvert = Gst.ElementFactory.make("nvvidconv", "nvvideoconvert")
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        parser = Gst.ElementFactory.make("h265parse", "parser")
        payloader = Gst.ElementFactory.make("rtph265pay", "payloader")
        sink = Gst.ElementFactory.make("udpsink", "sink")

        if not all([multicast_pipeline, appsrc, nvvideoconvert,
                    encoder, parser, payloader, sink]):
            print("Not all elements could be created.")
            return None

        

        multicast_pipeline.add(appsrc)
        multicast_pipeline.add(nvvideoconvert)
        multicast_pipeline.add(encoder)
        multicast_pipeline.add(parser)
        multicast_pipeline.add(payloader)
        multicast_pipeline.add(sink)


        if not (appsrc.link(nvvideoconvert) and
                nvvideoconvert.link(encoder) and encoder.link(parser) and
                parser.link(payloader) and payloader.link(sink)):
            print("Elements could not be linked.")
            return None

        appsrc.set_property("caps", Gst.Caps.from_string("video/x-raw,format=I420,width=1280,height=720,framerate=30/1"))
        encoder.set_property("bitrate", 10000)
        payloader.set_property("config-interval", 1)
        sink.set_property("host", self)
        sink.set_property("auto-multicast", True)
        sink.set_property("multicast-iface", "wlan0")
        sink.set_property("port", 5000)
        return multicast_pipeline, appsrc

    def push_data(appsrc, result_frame):

        i420_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2YUV_I420)
        # Convert frame to GstBuffer
        data = i420_frame.tobytes()
        # print("push-data: read new image: ", len(data))
        buffer = Gst.Buffer.new_allocate(None, len(data), None)
        buffer.fill(0, data)
        
        # Push buffer to appsrc
        retval = appsrc.emit("push-buffer", buffer)
        if retval != Gst.FlowReturn.OK:
            print("Error pushing buffer to appsrc")
            return False
    
        return True
        
class RealSense:
    # RealSense setup
    def setup_realsense(DEPTH_WIDTH, DEPTH_HEIGHT, 
                        COLOR_WIDTH, COLOR_HEIGHT, FPS, laser_power_set):
        
        camera_pipeline = rs.pipeline()
        camera_config = rs.config()

        # Get device product line for setting a supporting resolution 
        camera_pipeline_wrapper = rs.pipeline_wrapper(camera_pipeline)
        camera_pipeline_profile = camera_config.resolve(camera_pipeline_wrapper)
        camera_device = camera_pipeline_profile.get_device()
        device_product_line = str(camera_device.get_info(rs.camera_info.product_line))
        camera_config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
        camera_config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
        # try:
        #     camera_pipeline.start(camera_config)
        # except RuntimeError:
        #     camera_pipeline.stop()
        #     camera_pipeline.start(camera_config)
        # Configure emitter settings
        depth_sensor = camera_device.first_depth_sensor()
        
        # Check if the emitter is supported by the device
        if depth_sensor.supports(rs.option.emitter_enabled):
            # Enable emitter (1) or disable emitter (0)
            depth_sensor.set_option(rs.option.emitter_enabled, 1)  # Turn on emitter
            print("Emitter enabled.")

            # Optionally adjust laser power
            if depth_sensor.supports(rs.option.laser_power):
                # Set the laser power between 0 and max laser power
                laser_power = depth_sensor.get_option_range(rs.option.laser_power)
                depth_sensor.set_option(rs.option.laser_power, laser_power_set)  # Set to max power
                print(f"Laser power set to: {laser_power_set}.")

        camera_pipeline.start(camera_config)
        return camera_pipeline