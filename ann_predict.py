def func():
    import cv2
    import matplotlib.pyplot as plt
    import copy
    import numpy as np
    import json
    from torch import nn

    from src import model
    from src import util
    from src.body import Body
    from src.hand import Hand
    from ann import ANN_Model
    from torch.autograd import Variable
    import os
    import torch
    ann_path = "ANNModel.pth"
    ANN_Model=ANN_Model()
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    body_estimation = Body('model/body_pose_model.pth')
    hand_estimation = Hand('model/hand_pose_model.pth')
    data_name=0

    test_image = 'yolo_output/exp/abc.jpg'
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    normal=0
    hand=0
    not_normal=0
    not_normal_hand=0
    for a in range(subset.shape[0]):
        data_dict={}
        #print(a)
        for b in range(18):
            for c in range(candidate.shape[0]):
                if subset[a,b]==int(candidate[c,3]):
                    data_dict.setdefault(b,[candidate[c,0],candidate[c,1],candidate[c,2]])

                if subset[a,b]==-1:
                    data_dict.setdefault(b,[-1,-1,-1])


        data_list1=[]
        data_list2=[]
        data_list3=[]
        data_list4=[]
        data_list5=[]
        data_list6=[]
        data_list7=[]
        data_list8=[]
        data_list9=[]
        data_list10=[]
        data_list11=[]
        data_list12=[]
        data_list13=[]
        data_list14=[]
        data_list15=[]
        data_list16=[]
        data_list17=[]
        data_list18=[]

        for d in range(6):
            data_list1.append(data_dict[d][0])
        for e in range(6):
            data_list2.append(data_dict[e+6][0])
        for f in range(6):
            data_list3.append(data_dict[f+12][0])
        for g in range(6):
            data_list4.append(data_dict[g][1])
        for h in range(6):
            data_list5.append(data_dict[h+6][1])
        for i in range(6):
            data_list6.append(data_dict[i+12][1])
        data_lists1=np.vstack([data_list1,data_list2,data_list3,data_list4,data_list5,data_list6])

        for j in range(6):
            data_list7.append(data_dict[j][1])
        for k in range(6):
            data_list8.append(data_dict[k+6][1])
        for l in range(6):
            data_list9.append(data_dict[l+12][1])
        for m in range(6):
            data_list10.append(data_dict[m][2])
        for n in range(6):
            data_list11.append(data_dict[n+6][2])
        for o in range(6):
            data_list12.append(data_dict[o+12][2])
        data_lists2=np.vstack([data_list7,data_list8,data_list9,data_list10,data_list11,data_list12])

        for p in range(6):
            data_list13.append(data_dict[p][2])
        for q in range(6):
            data_list14.append(data_dict[q+6][2])
        for r in range(6):
            data_list15.append(data_dict[r+12][2])
        for s in range(6):
            data_list16.append(data_dict[s][0])
        for t in range(6):
            data_list17.append(data_dict[t+6][0])
        for u in range(6):
            data_list18.append(data_dict[u+12][0])
        data_lists3=np.vstack([data_list13,data_list14,data_list15,data_list16,data_list17,data_list18])
        data_list_all=np.stack((data_lists1,data_lists2,data_lists3),axis=0)


        ANN_Model.load_state_dict(torch.load(ann_path))
        data_teat=np.reshape(data_lists1,(-1,36))

        import torchvision

        anndata = torch.tensor(data_teat)
        outputs = ANN_Model(anndata.float())
        ef, predicted = torch.max(outputs,1)
        ef = ef.detach().numpy()
        predicted = np.array(predicted)
        if np.count_nonzero(data_lists1 == -1) >= 26:
            continue
        for i in predicted:
            if i==0:
                normal+=1
                #print("正常")
            elif i ==1:
                hand+=1
                #print("舉手")
            elif i ==2:
                not_normal_hand+=1
                #print("行動不便舉手")
            elif i ==3:
                not_normal+=1
                #print("行動不便")
        data_name+=1



    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # if is_left:
            # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            # plt.show()
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        # else:
        #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     print(peaks)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)

    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.savefig('./images/ann_predict.jpg')
    plt.close()
    #plt.show()
    return subset.shape[0],normal,hand,not_normal_hand,not_normal
if __name__ == "__main__":
    people,normal,hand,not_normal_hand,not_normal=func()
    print("總人數:", people, "  ")
    print("符合正常人數:", normal)
    print("舉手搭車人數:", hand)
    print("行動不便舉手人數:", not_normal_hand)
    print("行動不便人數:", not_normal)