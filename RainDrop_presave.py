import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import os

year = ['2012','2013','2014','2015','2016','2017','2018','2019']
year_test = ['2019']
test_list = 'C:/Users/--/AllRainDropUnetData/list_contest_Q18.txt'
# 테스트 리스트가 존재할 시 (단, 연도는 모두 동일해야한다 & year_test에 연도도 추가해야한다)
# 테스트 리스트가 존재하지 않으면 문자열을 아무렇게나 바꾼다.


predata_dir = 'C:/Users/--/RDR_data/'                    # 전처리 전 데이터 위치
save_dir = 'C:/Users/--/AllRainDropUnetData/'            # 전처리 후 데이터 저장 위치 & 참조 위치


def read_all_file(path):
    output = os.listdir(path)
    file_list = []
    for i in output:
        if 'DS' in i:
            continue
        if 'cmap' in i:
            continue
        if os.path.isdir(path+"/"+i):
            file_list.extend(read_all_file(path+"/"+i))
        elif os.path.isfile(path+"/"+i):
            file_list.append(path+"/"+i)

    return file_list                        # processing()에서 사용할 read함수입니다. 연도별 폴더 내 다른 파일을 제외하고 읽습니다.


def processing():                           # 데이터 전처리 processing() 함수
    for i in range(len(year)):
        time = year[i]
        for filename in read_all_file(predata_dir+time):
            ncfile = Dataset(filename)
            rain = (np.array(ncfile.variables["rain1h"]))
            rain[rain<0]=0
            rain[rain>120]=120
            ########################### 전처리 핵심
            rain = np.log10(rain+0.01)
            ########################### 대략 (-2 ~ 2.5)로 로그 전처리
            np.round(rain,5)

            year_save_dir = os.path.join(save_dir, time)
            if not os.path.exists(year_save_dir):
                os.makedirs(year_save_dir)
            np.save(save_dir+time+'/'+filename[-22:-9],rain)

            # [0, 120]으로 범위 고정 후, 로그 전처리를 하여 [연도별 폴더 - 시간 이름 파일]로 저장합니다.


def labeling(testlist):                             # 데이터 라벨링 labeling() 함수 : input = 4frames / label(output) = after 1frames
    if not os.path.exists(os.path.join(save_dir,'data')):
                os.makedirs(os.path.join(save_dir,'data'))
    if not os.path.exists(os.path.join(save_dir,'data','test')):
                os.makedirs(os.path.join(save_dir,'data','test'))
    if not os.path.exists(os.path.join(save_dir,'data','test','input')):
                os.makedirs(os.path.join(save_dir,'data','test','input'))
    if not os.path.exists(os.path.join(save_dir,'data','test','label')):
                os.makedirs(os.path.join(save_dir,'data','test','label'))
    if not os.path.exists(os.path.join(save_dir,'data','val')):
                os.makedirs(os.path.join(save_dir,'data','val'))
    if not os.path.exists(os.path.join(save_dir,'data','val','input')):
                os.makedirs(os.path.join(save_dir,'data','val','input'))
    if not os.path.exists(os.path.join(save_dir,'data','val','label')):
                os.makedirs(os.path.join(save_dir,'data','val','label'))
    if not os.path.exists(os.path.join(save_dir,'data','train')):
                os.makedirs(os.path.join(save_dir,'data','train'))
    if not os.path.exists(os.path.join(save_dir,'data','train','input')):
                os.makedirs(os.path.join(save_dir,'data','train','input'))
    if not os.path.exists(os.path.join(save_dir,'data','train','label')):
                os.makedirs(os.path.join(save_dir,'data','train','label'))

    testlist_exist = os.path.exists(testlist)
    # print(testlist_exist)
    testlist_arr = []
    if(testlist_exist):
        f = open(testlist,'r')
        while True:
            line = f.readline()
            if not line: break
            testlist_arr.append(line.strip())
        f.close()


    for i in range(len(year)):
        time = year[i]
        filelist = os.listdir(save_dir+time)
        if time in year_test:
            if(testlist_exist):
                for j in range(len(testlist_arr)):
                    filename = testlist_arr[j][0:4]+'-'+testlist_arr[j][4:6]+'-'+testlist_arr[j][6:8]+'_'+testlist_arr[j][8:10]+'.npy'
                    index = filelist.index(filename)
                    a=[]
                    for k in range(4):
                        put = np.load(save_dir+time+'/'+filelist[index-4+k])
                        a.append(put)
                    np.save(save_dir+'data/test/input/'+filename,a)
                    np.save(save_dir+'data/test/label/'+filename,np.load(save_dir+time+'/'+filename))
            else:
                for j in range(len(filelist)-4):
                    a=[]
                    for k in range(4):
                        put = np.load(save_dir+time+'/'+filelist[j+k])
                        a.append(put)
                    np.save(save_dir+'data/test/input/'+filelist[j+4],a)
                    np.save(save_dir+'data/test/label/'+filelist[j+4],np.load(save_dir+time+'/'+filelist[j+4]))
        else :
            for j in range(len(filelist)-4):
                r = np.random.rand()
                a=[]
                for k in range(4):
                    put = np.load(save_dir+time+'/'+filelist[j+k])
                    a.append(put)
                if r>0.9 :
                    #validation dataset이 필요한 경우, continue를 주석처리, 그 밑 주석해제
                    continue
                    # np.save(save_dir+'data/val/input/'+filelist[j+4],a)
                    # np.save(save_dir+'data/val/label/'+filelist[j+4],np.load(save_dir+time+'/'+filelist[j+4]))
                else:
                    np.save(save_dir+'data/train/input/'+filelist[j+4],a)
                    np.save(save_dir+'data/train/label/'+filelist[j+4],np.load(save_dir+time+'/'+filelist[j+4]))

def test_image_save():
    filelist = os.listdir(save_dir+'data/test/label/')
    for j in range(len(filelist)):
        reading = np.load(save_dir+'data/test/label/'+filelist[j])
        if not os.path.exists(os.path.join(save_dir,'data','test_image')):
                os.makedirs(os.path.join(save_dir,'data','test_image'))
        plt.imsave(save_dir+'data/test_image/'+filelist[j]+'.png',reading,cmap='jet',vmin=-2.5,vmax=2.5)

# processing()        #전처리&저장
labeling(test_list) #input/label 분류
# test_image_save()   #test_image 저장
