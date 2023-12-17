import numpy as np

import time
import cv2

from config import cfg
from lib.models.movenet_mobilenetv2 import MoveNet 
import torch.nn as nn
import torch
import torch.nn.functional as F

from lib.task.task_tools import movenetDecode


# class Task():
#     def __init__(self, cfg, model):

#         self.cfg = cfg

#         if self.cfg['GPU_ID'] != '' :
#             self.device = torch.device("cuda")
#         else:
#             self.device = torch.device("cpu")

#         self.model = model.to(self.device)

#         ############################################################
#         # loss

#         # scheduler

#     def predict(self, data_loader):
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)

#         self.model.eval()
#         correct = 0

#         with torch.no_grad():

#             for (img, img_name) in data_loader:

#                 # if "yoga_img_483" not in img_name[0]:
#                 #     continue

#                 # print(img.shape, img_name)
#                 img = img.to(self.device)

#                 output = self.model(img)
#                 #print(len(output))
                


#                 pre = movenetDecode(output, None, mode='output')
#                 print(pre)


#                 basename = os.path.basename(img_name[0])
#                 img = np.transpose(img[0].cpu().numpy(),axes=[1,2,0])
#                 img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#                 h,w = img.shape[:2]

#                 for i in range(len(pre[0])//2):
#                     x = int(pre[0][i*2]*w)
#                     y = int(pre[0][i*2+1]*h)
#                     cv2.circle(img, (x, y), 3, (255,0,0), 2)

#                 cv2.imwrite(os.path.join(save_dir,basename), img)
                

#                 ## debug
#                 heatmaps = output[0].cpu().numpy()[0]
#                 centers = output[1].cpu().numpy()[0]
#                 regs = output[2].cpu().numpy()[0]
#                 offsets = output[3].cpu().numpy()[0]

#                 #print(heatmaps.shape)
#                 hm = cv2.resize(np.sum(heatmaps,axis=0),(192,192))*255
#                 cv2.imwrite(os.path.join(save_dir,basename[:-4]+"_heatmaps.jpg"),hm)
#                 img[:,:,0]+=hm
#                 cv2.imwrite(os.path.join(save_dir,basename[:-4]+"_img.jpg"), img)
#                 cv2.imwrite(os.path.join(save_dir,basename[:-4]+"_center.jpg"),cv2.resize(centers[0]*255,(192,192)))
#                 cv2.imwrite(os.path.join(save_dir,basename[:-4]+"_regs0.jpg"),cv2.resize(regs[0]*255,(192,192)))
                


#     def label(self, data_loader, save_dir):
#         self.model.eval()
        

#         txt_dir = os.path.join(save_dir, 'txt')
#         show_dir = os.path.join(save_dir, 'show')

#         with torch.no_grad():

#             for (img, img_path) in data_loader:
#                 #print(img.shape, img_path)
#                 img_path = img_path[0]
#                 basename = os.path.basename(img_path)

#                 img = img.to(self.device)

#                 output = self.model(img)
#                 #print(len(output))
                


#                 pre = movenetDecode(output, None, mode='output')[0]
#                 #print(pre)
#                 with open(os.path.join(txt_dir,basename[:-3]+'txt'),'w') as f:
#                     f.write("7\n")
#                     for i in range(len(pre)//2):
#                         vis = 2
#                         if pre[i*2]==-1:
#                             vis=0
#                         line = "%f %f %d\n" % (pre[i*2], pre[i*2+1], vis)
#                         f.write(line)

                

#                 img = np.transpose(img[0].cpu().numpy(),axes=[1,2,0])
#                 img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#                 h,w = img.shape[:2]

#                 for i in range(len(pre)//2):
#                     x = int(pre[i*2]*w)
#                     y = int(pre[i*2+1]*h)
#                     cv2.circle(img, (x, y), 3, (255,0,0), 2)

#                 cv2.imwrite(os.path.join(show_dir,basename), img)
                

#                 #b
                
#     def modelLoad(self,model_path, data_parallel = False):
#         self.model.load_state_dict(nn.load(model_path), strict=True)
        
#         if data_parallel:
#             self.model = nn.DataParallel(self.model)

# class TensorImageTest():

#     def __init__(self, data_labels, img_dir, img_size):
#         self.data_labels = data_labels
#         self.img_dir = img_dir
#         self.img_size = img_size


#         self.interp_methods = cv2.INTER_LINEAR


#     def __getitem__(self, index):

#         img_name = self.data_labels[index]

#         img = cv2.imread(img_name, cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         img = cv2.resize(img, (self.img_size, self.img_size),
#                                 interpolation=self.interp_methods)


#         img = img.astype(np.float32)
#         img = np.transpose(img,axes=[2,0,1])


#         return img, img_name


def main():
    
    cap = cv2.VideoCapture(-0)
    
    model = MoveNet(num_classes=cfg["num_classes"],
                width_mult=cfg["width_mult"],
                mode='train')
    if cfg['GPU_ID'] != '' :
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    model.load_state_dict(torch.load('output/e118_valacc0.79805.pth'), strict=True)
        
    
    # if data_parallel:
    #     self.model = torch.nn.DataParallel(self.model)
    model.eval()

    while True:
        ret, frame = cap.read()
        # fps calculation
        fps = 0
        prev_frame_time = time.time()
        new_frame_time = 0  
        # cv2.imshow("frame", frame)

        if ret:
            # img = cv2.imread(frame, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (cfg['img_size'], cfg['img_size']),
                        interpolation=cv2.INTER_LINEAR)
            # img = img.to(device)
            img_ = img.astype(np.float32)
            img_ = np.transpose(img,axes=[2,0,1])

            img_ = img_.reshape(1, 3, 192, 192)
            img_ = torch.tensor(img_).to(device) #.unsqueeze_(0)

            output = model.forward(img_)
            # output = output.unsqueeze_(0)
            pre = movenetDecode(output, None, mode='output')


            # basename = os.path.basename(img_name[0])
            # img = np.transpose(img[0].cpu().numpy(),axes=[1,2,0])
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            h,w = img.shape[:2]
            img = cv2.resize(frame, (cfg['img_size'], cfg['img_size']),
                        interpolation=cv2.INTER_LINEAR)
            for i in range(len(pre[0])//2):
                x = int(pre[0][i*2]*w)
                y = int(pre[0][i*2+1]*h)
                cv2.circle(img, (x, y), 3, (255,0,0), 2)
            # img = cv2.resize(img, (cfg['img_size'], cfg['img_size']),
            #             interpolation=cv2.INTER_LINEAR)
            # FPS calculation
            # img = cv2.resize(img, frame.shape(),
            #             interpolation=cv2.INTER_LINEAR)    

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(img, f'FPS: {fps:.2f}', (60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()