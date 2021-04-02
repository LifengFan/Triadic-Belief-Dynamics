import shutil
import torch
import torch.utils.data
import os


def load_best_checkpoint(model,path):
    if path:
       checkpoint_dir=path
       best_model_file=os.path.join(checkpoint_dir)
       if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
       if os.path.isfile(best_model_file):
            print("====> loading best model {}".format(best_model_file))
            checkpoint=torch.load(best_model_file)
            #print(checkpoint['state_dict'].keys())
            model_dict = model.state_dict()
            pretrained_model = checkpoint['state_dict']
            pretrained_dict = {}
            for k, v in pretrained_model.items():
                if k[len('module.'):] in model_dict:
                    pretrained_dict[k[len('module.'):]] = v
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            model.cuda()
            print("===> loaded best model {} (epoch {})".format(best_model_file,checkpoint['epoch']))
            return  model
       else:
           print('===> no best model found at {}'.format(best_model_file))
    else:
        return None


def save_checkpoint(state,is_best,directory, version):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file=os.path.join(directory,'checkpoint_'+version+'.pth')
    best_model_file=os.path.join(directory,'model_best.pth')

    torch.save(state,checkpoint_file)

    if is_best:

        shutil.copyfile(checkpoint_file,best_model_file)

class Path():
    def __init__(self, mode):
        if mode=='home':
            self.home_path = '/home/lfan/Dropbox/Projects/NIPS20/'
            self.home_path2 = '/media/lfan/HDD/NIPS20/'
            self.data_path=self.home_path + 'data/'
            self.data_path2=self.home_path2+'data/'
            self.img_path=self.home_path+'annotations/'
            self.save_root=self.home_path2
            self.save_path=self.save_root+'BeamSearch_home'
            self.annotation_path=self.home_path + 'reformat_annotation/'
            self.reannotation_path = self.home_path + 'regenerate_annotation/'

            self.init_cps='/media/lfan/HDD/NIPS20/data/init_cps/CPS_NEW_0601.p'
            self.stat_path=self.home_path + 'data/stat/'
            self.attmat_path=self.home_path+'data/record_attention_matrix/'
            self.cate_path=self.home_path2+ 'data/track_cate/'
            self.tracker_bbox=self.home_path2+'data/tracker_record_bbox/'
            self.battery_bbox=self.home_path2+'data/record_bbox/'
            self.obj_bbox=self.home_path2+'data/post_neighbor_smooth_newseq/'
            self.ednet_path=self.home_path + 'model/ednet_tuned_best.pth'
            self.atomic_path=self.home_path + 'model/atomic_best.pth'
            self.seg_label=self.home_path + 'data/segment_labels/'
            self.feature_single=self.home_path2+'data/feature_single/'
            self.save_event_score='/media/lfan/HDD/NIPS20/data/EVENT_SCORE/'
            self.mind_model_path=self.home_path+'model/model_best_event_memory.pth'

        elif mode=='azure':
            pass
        elif mode=='thor':
            pass
    def update(self, name, path):
        self.name=path

def main():

    pass


if __name__=='__main__':
    main()