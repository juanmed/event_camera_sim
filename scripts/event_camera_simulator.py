
import torch
import torch.nn as nn
import numpy as np


class EventCameraSim(nn.Module):

    def __init__(self, initial_image, initial_time = 0.0, C = 0.15):
        super(EventCameraSim, self).__init__()
        """
        Initialize simulator. Requires an initial gray scale image.
        initial_image m by n np.array, is initial image
        initial_time float timestamp corresponding to the initial image
        C event generation threshold
        """
        self.C = C
        assert(initial_image.shape[0] > 0)
        assert(initial_image.shape[1] > 0)
        self.height = initial_image.shape[0]
        self.width = initial_image.shape[1]
        self.reference_values = self.safe_log(torch.from_numpy(initial_image.copy().astype(np.float32)))
        self.It_array =  self.reference_values.clone() #self.safe_log(torch.from_numpy(initial_image.copy().astype(np.float32)))
        self.t = initial_time

        # tensor containing initial times of all pixels
        self.ts = torch.ones_like(self.It_array) * self.t

        self.tol = 1e-6  # minimum brightness change for which
                         # an event could be generated
        self.initial_image = np.zeros_like(initial_image) # initial_image.copy()

    def forward(self,x,time):
        # convert from numpy to torch
        x = torch.from_numpy(x.astype(np.float32))
        x = self.safe_log(x)
        assert(x.shape == self.It_array.shape)

        delta_t = time - self.t

        # get number of events deltaI/C per pixel, together with their polarity (sign)
        deltaI = x - self.It_array
        number_events = torch.floor(torch.abs(torch.div(deltaI, self.C)))
        polarities = torch.where(number_events > 0., torch.sign(deltaI),torch.Tensor([0]))
        slope = torch.div(deltaI,delta_t)
        channels = x.shape[2]
        per_channel_events = list()

        for channel in range(channels):
            print("Ch: {}".format(channel))
            #print(deltaI[:,:,channel])
            print(number_events[:,:,channel] * polarities[:,:,channel])
            #print(polarities[:,:,channel])
            #print(torch.max(number_events[:,:,channel]),torch.min(number_events))
            img_interpolation = torch.ones((x.shape[0],x.shape[1],int(torch.max(number_events[:,:,channel]).item())), dtype = self.It_array.dtype)
            pol = torch.repeat_interleave(polarities[:,:,channel].unsqueeze(2),img_interpolation.shape[-1],dim=2).type(img_interpolation.dtype)
            #print(pol)#.shape, img_interpolation.shape)
            img_interpolation =torch.mul(img_interpolation, pol)[:,:,:]* torch.arange(1, img_interpolation.shape[-1]+1).type(pol.dtype)*self.C

            #time_events = torch.ones((x.shape[0],x.shape[1],int(torch.max(number_events).item())))
            #time_events[:,:,:] = time_events[:,:,:] * torch.arange(1, time_events.shape[-1]+1)*C/m
            slopes = torch.repeat_interleave(slope[:,:,channel].unsqueeze(2),img_interpolation.shape[-1],dim=2).type(img_interpolation.dtype)
            time_all_events = torch.zeros_like(img_interpolation)
            time_all_events = torch.where( (torch.abs(pol)>0.), torch.div(img_interpolation,slopes), torch.Tensor([float('nan')]))
            #print(time_all_events)
            It_arrays = torch.repeat_interleave(self.It_array[:,:,channel].unsqueeze(2),img_interpolation.shape[-1],dim=2).type(img_interpolation.dtype)
            xs = torch.repeat_interleave(x[:,:,channel].unsqueeze(2),img_interpolation.shape[-1],dim=2).type(img_interpolation.dtype)
            val1 = (pol > 0.) &  ((img_interpolation + It_arrays) < xs)
            #print((pol>0.) & ((img_interpolation + It_arrays) >= xs) )
            time_events = torch.where( val1, time_all_events, torch.Tensor([float('nan')]) )
            val2 = (pol < 0.) & ((img_interpolation + It_arrays) > xs)
            time_events = torch.where( val2, time_all_events, time_events)
            #print(img_interpolation + It_arrays)
            #print(xs)
            print(time_events)
            per_channel_events.append(time_events)



        # intensity increase generates positive event, otherwise negative event
        self.It_array = x   
        return x
    
    def safe_log(self,I):
        """
        Return the logarithm of input even if some of its
        elements are zero, by summing a small number before
        the computation
        """
        eps = 0.001
        return torch.log(I + eps)

def events_to_frame(events):

    if init:
        img = np.zeros((height, width,3), dtype=np.uint8)
        for event in events:
            channel = 0 if event[3] else 2
            x = event[1]
            y = event[0]
            img[x,y,channel] = 255
        return img
    else:
        print("Event Generator not initialized. Initialize parameter first")
        sys.exit(0)

if __name__ == '__main__':

    import os
    import random
    import cv2
    import time

    DATASET_DIR = '../../'
    TRAIN_DIR = ''  
    file = "cheetah_cut.mkv"
    train_files = os.listdir(os.path.join(DATASET_DIR, TRAIN_DIR))
    video_file = random.sample(train_files, 1)[0]
    video_dir = os.path.join(DATASET_DIR, TRAIN_DIR, file)
    video = cv2.VideoCapture(video_dir)

    frame_number = 0
    period = 1./30.
    save_event_video = False
    init = False

    while(video.isOpened()):
        try:
            ret, frame = video.read()
            frame = cv2.resize(frame,(5,5))
        except:
            break
        #for each frame, obtain face features
        if frame is None:
            break

        #works with grayscale images
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp = frame_number*period

        if not init:
            init = True # only do this once
            last_pub_event_timestamp = timestamp

            sim = EventCameraSim(frame, timestamp)

            height = frame.shape[0]
            width = frame.shape[1]

            if save_event_video:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')    
                out = cv2.VideoWriter("black.avi",fourcc, 30.0, (width,height))                    

            print("initialization complete, sensor size (w:{} h:{})".format(width, height))

        else:
            # compute events for this frame
            current_events = sim(frame, timestamp)
            events = current_events
            
            # DO SOMETHING WITH EVENTS
            #print("Frame: {} events: {} time: {:.2f}".format(frame_number, len(self.events), t2-t1))
            
            if save_event_video:
                #event_img = events_to_frame(events)
                out.write(np.uint8(events.numpy()))

        frame_number = frame_number + 1

    if save_event_video and init:
        print("Video saved")
        out.release()




