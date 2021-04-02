import pickle
from train_atomic_node_only import *
import argparse
import torch
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import utils_atomic


def get_model_output(test_loader, model):

    features = np.empty((0, 108*5))
    labels = np.empty(0)
    for i, (head_batch, pos_batch, attmat_batch, index, atomic_label_batch) in enumerate(test_loader):

        if args.cuda:
            heads = (torch.autograd.Variable(head_batch)).cuda()
            poses = (torch.autograd.Variable(pos_batch)).cuda()
            attmat_gt = (torch.autograd.Variable(attmat_batch)).cuda()

        _, feature_embed = model(heads, poses, attmat_gt)  # [N, 6, 1,1,1]
        feature_embed = feature_embed.data.cpu().numpy()
        features = np.vstack([features, feature_embed])
        labels = np.append(labels, atomic_label_batch)
    return features, labels

def get_data_input(args):
    with open('fine_tune_input_3.p', 'rb') as f:
        train_loader, validate_loader, test_loader = pickle.load(f)

    model = Atomic_node_only_lstm_first_view()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # use multi-gpu

    if args.cuda and torch.cuda.device_count() > 1:
        print("Now Using ", len(args.device_ids), " GPUs!")

        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0]).cuda()


    elif args.cuda:
        model = model.cuda()


    if args.load_best_checkpoint:
        loaded_checkpoint = utils_atomic.load_best_checkpoint(args, model, optimizer, path=args.resume)

        if loaded_checkpoint:
            args, best_epoch_acc, avg_epoch_acc, model, optimizer = loaded_checkpoint

    if args.load_last_checkpoint:
        loaded_checkpoint = utils_atomic.load_last_checkpoint(args, model, optimizer, path=args.resume,
                                                              version=args.model_load_version)

        if loaded_checkpoint:
            args, best_epoch_acc, avg_epoch_acc, model, optimizer = loaded_checkpoint

    model.eval()
    train_x, train_y = get_model_output(train_loader, model)
    validate_x, validate_y = get_model_output(validate_loader, model)
    test_x, test_y = get_model_output(test_loader, model)
    with open('random_forest_input.p', 'wb') as f:
        joblib.dump([train_x, train_y, validate_x, validate_y, test_x, test_y], f)

def train():
    with open('random_forest_input.p', 'wb') as f:
        train_x, train_y, validate_x, validate_y, test_x, test_y = joblib.load(f)

    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(train_x, train_y)

    y_pred = clf.predict(test_x)
    print(metrics.classification_report(test_y, y_pred, digits=3))


def parse_arguments():

    project_name = 'train_atomic_node_only_lstm'
    parser = argparse.ArgumentParser(description=project_name)
    parser.add_argument('--project-name', default=project_name, help='project name')

    # path settings
    parser.add_argument('--project-root', default='.', help='project root path')
    parser.add_argument('--tmp-root', default='.', help='checkpoint path')
    parser.add_argument('--log-root', default='./fine_tune_cptk/', help='log files path')
    parser.add_argument('--resume', default='.',help='path to the latest checkpoint')
    parser.add_argument('--save-test-res', default='./fine_tune_cptk/',help='path to save test metrics')

    # optimization options
    parser.add_argument('--load-last-checkpoint', default=False, help='To load the last checkpoint as a starting point for model training')
    parser.add_argument('--load-best-checkpoint', default=False,help='To load the best checkpoint as a starting point for model training')
    parser.add_argument('--batch-size', type=int, default=56, help='Input batch size for training (default: 10)')
    parser.add_argument('--use-cuda', default=True, help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default : 10)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Index of epoch to start (default : 0)')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate (default : 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='Learning rate decay factor (default : 0.6)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default : 0.9)')
    parser.add_argument('--visdom', default=False, help='use visdom to visualize loss curve')

    parser.add_argument('--device-ids', default=[0,1], help='gpu ids')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()


    #    if args.visdom:
    #        vis = visdom.Visdom()
    #        assert vis.check_connection()

    get_data_input(args)