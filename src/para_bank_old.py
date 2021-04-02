import random
import os
import pickle
import os.path as op
import numpy as np

# def generate_para_bank(args): # --round 1
#     #  build parameter bank
#     para_bank=[]
#     for topN in [5]:
#         for lambda_2 in [0.1, 1, 10]:
#             for lambda_3 in [0.1, 1, 10]:
#                 for lambda_4 in [0.1, 1, 10]:
#                     for lambda_5 in [0.1, 1, 10]:
#                         for lambda_6 in [0.1, 1, 10]:
#                             for beta_1 in [0.1, 1, 10]:
#                                 for beta_2 in [0.1, 1, 10]:
#                                     for beta_3 in [0.1, 1, 10]:
#                                         for hist_bin in [10]:
#                                             for search_N_cp in [5]:
#                                                 para_bank.append({'topN':topN, 'lambda_2':lambda_2, 'lambda_3':lambda_3,
#                                                                   'lambda_4':lambda_4, 'lambda_5':lambda_5,'lambda_6':lambda_6,
#                                                                   'beta_1':beta_1, 'beta_2':beta_2, 'beta_3':beta_3, 'hist_bin':hist_bin, 'search_N_cp':search_N_cp})
#
#     random.seed(0)
#     random.shuffle(para_bank)
#     print("Parameter Bank Done! There are {} sets of parameters totally.".format(len(para_bank)))
#     print("="*74)
#     with open(op.join(args.save_path, 'para_bank.p'), 'wb') as f:
#         pickle.dump(para_bank, f)
#
#     return para_bank

# def generate_para_bank(args): --round 2
#     #  build parameter bank
#     para_bank=[]
#     for topN in [5]:
#         for lambda_2 in [0.1, 0.01]:
#             for lambda_3 in [0.1, 0.01]:
#                 for lambda_4 in [10, 20]:
#                     for lambda_5 in [0.1, 1, 10]:
#                         for lambda_6 in [0.1, 1, 10]:
#                             for beta_1 in [0.1, 0.01]:
#                                 for beta_2 in [0.1, 1, 10]:
#                                     for beta_3 in [0.1, 0.01]:
#                                         for hist_bin in [10]:
#                                             for search_N_cp in [5]:
#                                                 para_bank.append({'topN':topN, 'lambda_2':lambda_2, 'lambda_3':lambda_3,
#                                                                   'lambda_4':lambda_4, 'lambda_5':lambda_5,'lambda_6':lambda_6,
#                                                                   'beta_1':beta_1, 'beta_2':beta_2, 'beta_3':beta_3, 'hist_bin':hist_bin, 'search_N_cp':search_N_cp})
#
#     random.seed(0)
#     random.shuffle(para_bank)
#     print("Parameter Bank Done! There are {} sets of parameters totally.".format(len(para_bank)))
#     print("="*74)
#     with open(op.join(args.save_path, 'para_bank.p'), 'wb') as f:
#         pickle.dump(para_bank, f)
#
#     return para_bank

# def generate_para_bank(args): # --round 3
#     #  build parameter bank
#     para_bank=[]
#     for topN in [5]:
#         for lambda_2 in [0.1, 0.5]:
#             for lambda_3 in [0.1, 0.5]:
#                 for lambda_4 in [10, 30]:
#                     for lambda_5 in [0.1, 1, 10]:
#                         for lambda_6 in [0.1, 1, 10]:
#                             for beta_1 in [0.1, 0.5, 0.01]:
#                                 for beta_2 in [0.1, 1, 10]:
#                                     for beta_3 in [0.1, 0.5, 0.01]:
#                                         for hist_bin in [10]:
#                                             for search_N_cp in [5]:
#                                                 para_bank.append({'topN':topN, 'lambda_2':lambda_2, 'lambda_3':lambda_3,
#                                                                   'lambda_4':lambda_4, 'lambda_5':lambda_5,'lambda_6':lambda_6,
#                                                                   'beta_1':beta_1, 'beta_2':beta_2, 'beta_3':beta_3, 'hist_bin':hist_bin, 'search_N_cp':search_N_cp})
#
#     random.seed(0)
#     random.shuffle(para_bank)
#     print("Parameter Bank Done! There are {} sets of parameters totally.".format(len(para_bank)))
#     print("="*74)
#     with open(op.join(args.save_path, 'para_bank.p'), 'wb') as f:
#         pickle.dump(para_bank, f)
#
#     return para_bank

# def generate_para_bank(args): # --round 4
#     #  build parameter bank
#     para_bank=[]
#     for topN in [5]:
#         for lambda_2 in [0.3, 0.1, 0.05]:
#             for lambda_3 in [0.3, 0.1, 0.05]:
#                 for lambda_4 in [5, 10, 15]:
#                     for lambda_5 in [0.5, 1, 5]:
#                         for lambda_6 in [0.5, 1, 5]:
#                             for beta_1 in [0.1, 0.5, 0.05]:
#                                 for beta_2 in [0.5, 1, 5]:
#                                     for beta_3 in [0.1, 0.5, 0.05]:
#                                         for hist_bin in [10]:
#                                             for search_N_cp in [5]:
#                                                 para_bank.append({'topN':topN, 'lambda_2':lambda_2, 'lambda_3':lambda_3,
#                                                                   'lambda_4':lambda_4, 'lambda_5':lambda_5,'lambda_6':lambda_6,
#                                                                   'beta_1':beta_1, 'beta_2':beta_2, 'beta_3':beta_3, 'hist_bin':hist_bin, 'search_N_cp':search_N_cp})
#
#     random.seed(0)
#     random.shuffle(para_bank)
#     print("Parameter Bank Done! There are {} sets of parameters totally.".format(len(para_bank)))
#     print("="*74)
#     with open(op.join(args.save_path, 'para_bank.p'), 'wb') as f:
#         pickle.dump(para_bank, f)
#
#     return para_bank

def generate_para_bank_0525(args):
    #  build parameter bank
    para_bank=[]
    for topN in [5]:
        for lambda_1 in [5, 10, 20]:
            for lambda_2 in [0.1, 1]:
                for lambda_3 in [0.1, 1]:
                    for lambda_4 in [10, 20]:
                        for lambda_5 in [30, 10]:
                                for beta_1 in [1, 5]:
                                        for beta_3 in [0.1, 1]:
                                            for gamma_1 in [10, 20]:
                                                for search_N_cp in [8]:
                                                    para_bank.append({'topN':topN, 'lambda_1':lambda_1,'lambda_2':lambda_2, 'lambda_3':lambda_3,
                                                                      'lambda_4':lambda_4, 'lambda_5':lambda_5,
                                                                      'beta_1':beta_1, 'beta_3':beta_3, 'gamma_1':gamma_1, 'search_N_cp':search_N_cp})

    random.seed(0)
    random.shuffle(para_bank)
    print("Parameter Bank Done! There are {} sets of parameters totally.".format(len(para_bank)))
    print("="*74)
    with open(op.join(args.save_path, 'para_bank.p'), 'wb') as f:
        pickle.dump(para_bank, f)

    return para_bank

def generate_para_bank_0526(args):
    #  build parameter bank
    para_bank=[]
    for topN in [3]:
        for lambda_1 in [1, 5]:
            for lambda_2 in [0.1, 0.5]:
                for lambda_3 in [0.1, 0.5]:
                    for lambda_4 in [10, 15]:
                        for lambda_5 in [20]:
                                for beta_1 in [5, 10]:
                                        for beta_3 in [0.1, 0.5]:
                                            for gamma_1 in [10, 15, 20]:
                                                for search_N_cp in [5]:
                                                    para_bank.append({'topN':topN, 'lambda_1':lambda_1,'lambda_2':lambda_2, 'lambda_3':lambda_3,
                                                                      'lambda_4':lambda_4, 'lambda_5':lambda_5,
                                                                      'beta_1':beta_1, 'beta_3':beta_3, 'gamma_1':gamma_1, 'search_N_cp':search_N_cp})

    random.seed(0)
    random.shuffle(para_bank)
    print("Parameter Bank Done! There are {} sets of parameters totally.".format(len(para_bank)))
    print("="*74)
    with open(op.join(args.save_path, 'para_bank.p'), 'wb') as f:
        pickle.dump(para_bank, f)

    return para_bank

def generate_para_bank_0527v2(args):
    #  build parameter bank
    para_bank=[]
    for topN in [3]:
        for lambda_1 in [1, 5]:
            for lambda_2 in [0.1, 0.5]:
                for lambda_3 in [0.1, 0.5]:
                    for lambda_4 in [10, 15]:
                        for lambda_5 in [20]:
                                for beta_1 in [5, 10]:
                                        for beta_3 in [0.1, 0.5]:
                                            for gamma_1 in [10, 15, 20]:
                                                for search_N_cp in [5]:
                                                    para_bank.append({'topN':topN, 'lambda_1':lambda_1,'lambda_2':lambda_2, 'lambda_3':lambda_3,
                                                                      'lambda_4':lambda_4, 'lambda_5':lambda_5,
                                                                      'beta_1':beta_1, 'beta_3':beta_3, 'gamma_1':gamma_1, 'search_N_cp':search_N_cp})

    random.seed(0)
    random.shuffle(para_bank)
    print("Parameter Bank Done! There are {} sets of parameters totally.".format(len(para_bank)))
    print("="*74)
    with open(op.join(args.save_path, 'para_bank.p'), 'wb') as f:
        pickle.dump(para_bank, f)

    return para_bank


# [{'topN': 3, 'lambda_1': 5, 'lambda_2': 0.5, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 15, 'search_N_cp': 5}

#0525
# seg
# [763.1666666666666]
# [{'topN': 5, 'lambda_1': 5, 'lambda_2': 0.1, 'lambda_3': 0.1, 'lambda_4': 20, 'lambda_5': 10, 'beta_1': 5, 'beta_3': 1, 'gamma_1': 20, 'search_N_cp': 8}]
# event
# [912.2857142857143]
# [{'topN': 5, 'lambda_1': 10, 'lambda_2': 1, 'lambda_3': 1, 'lambda_4': 10, 'lambda_5': 30, 'beta_1': 5, 'beta_3': 0.1, 'gamma_1': 10, 'search_N_cp': 8}]
# all
# [2126.0333333333333]
# [{'topN': 5, 'lambda_1': 10, 'lambda_2': 1, 'lambda_3': 0.1, 'lambda_4': 10, 'lambda_5': 30, 'beta_1': 5, 'beta_3': 0.1, 'gamma_1': 20, 'search_N_cp': 8}]

# 0526 v1
# seg
# [788.4333333333333, 789.2, 790.7666666666667]
# [{'topN': 3, 'lambda_1': 5, 'lambda_2': 0.1, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 5, 'beta_3': 0.1, 'gamma_1': 20, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 1, 'lambda_2': 0.1, 'lambda_3': 0.5, 'lambda_4': 15, 'lambda_5': 20, 'beta_1': 5, 'beta_3': 0.1, 'gamma_1': 15, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 5, 'lambda_2': 0.1, 'lambda_3': 0.1, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 5, 'beta_3': 0.1, 'gamma_1': 20, 'search_N_cp': 5}]
# event
# [1269.0666666666666, 1269.0666666666666, 1269.5]
# [{'topN': 3, 'lambda_1': 1, 'lambda_2': 0.5, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 10, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 5, 'lambda_2': 0.5, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 10, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 1, 'lambda_2': 0.1, 'lambda_3': 0.5, 'lambda_4': 15, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 10, 'search_N_cp': 5}]
# all
# [2117.0666666666666, 2124.2, 2133.4666666666667]
# [{'topN': 3, 'lambda_1': 5, 'lambda_2': 0.5, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 15, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 5, 'lambda_2': 0.1, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 10, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 1, 'lambda_2': 0.5, 'lambda_3': 0.5, 'lambda_4': 15, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.1, 'gamma_1': 15, 'search_N_cp': 5}]

# seg
# [1691.0, 1705.0, 1708.5]
# [{'topN': 6, 'lambda_1': 3, 'lambda_2': 0.3, 'lambda_3': 0.3, 'lambda_4': 6, 'lambda_5': 40, 'beta_1': 10, 'beta_3': 0.3, 'gamma_1': 15, 'search_N_cp': 5}
#  {'topN': 6, 'lambda_1': 5, 'lambda_2': 0.3, 'lambda_3': 0.3, 'lambda_4': 6, 'lambda_5': 40, 'beta_1': 8, 'beta_3': 0.3, 'gamma_1': 8, 'search_N_cp': 5}
#  {'topN': 6, 'lambda_1': 5, 'lambda_2': 0.5, 'lambda_3': 0.3, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 8, 'beta_3': 0.8, 'gamma_1': 15, 'search_N_cp': 5}]
# event
# [1174.5, 1185.68, 1187.0]
# [{'topN': 6, 'lambda_1': 3, 'lambda_2': 0.3, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.8, 'gamma_1': 18, 'search_N_cp': 5}
#  {'topN': 6, 'lambda_1': 5, 'lambda_2': 0.3, 'lambda_3': 0.3, 'lambda_4': 6, 'lambda_5': 40, 'beta_1': 8, 'beta_3': 0.3, 'gamma_1': 8, 'search_N_cp': 5}
#  {'topN': 6, 'lambda_1': 3, 'lambda_2': 0.3, 'lambda_3': 0.3, 'lambda_4': 6, 'lambda_5': 40, 'beta_1': 10, 'beta_3': 0.3, 'gamma_1': 15, 'search_N_cp': 5}]
# all
# [2878.0, 2889.0833333333335, 2890.68]
# [{'topN': 6, 'lambda_1': 3, 'lambda_2': 0.3, 'lambda_3': 0.3, 'lambda_4': 6, 'lambda_5': 40, 'beta_1': 10, 'beta_3': 0.3, 'gamma_1': 15, 'search_N_cp': 5}
#  {'topN': 6, 'lambda_1': 3, 'lambda_2': 0.3, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.8, 'gamma_1': 18, 'search_N_cp': 5}
#  {'topN': 6, 'lambda_1': 5, 'lambda_2': 0.3, 'lambda_3': 0.3, 'lambda_4': 6, 'lambda_5': 40, 'beta_1': 8, 'beta_3': 0.3, 'gamma_1': 8, 'search_N_cp': 5}]

# seg
# [1195.5, 1622.764705882353, 1661.421052631579]
# [{'topN': 3, 'lambda_1': 5, 'lambda_2': 1, 'lambda_3': 0.1, 'lambda_4': 10, 'lambda_5': 30, 'beta_1': 10, 'beta_3': 0.1, 'gamma_1': 20, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 5, 'lambda_2': 1, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 30, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 20, 'search_N_cp': 5}
#  {'topN': 5, 'lambda_1': 5, 'lambda_2': 1, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 20, 'search_N_cp': 5}]
# event
# [340.0, 1155.3529411764705, 1186.7368421052631]
# [{'topN': 3, 'lambda_1': 5, 'lambda_2': 1, 'lambda_3': 0.1, 'lambda_4': 10, 'lambda_5': 30, 'beta_1': 10, 'beta_3': 0.1, 'gamma_1': 20, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 5, 'lambda_2': 1, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 30, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 20, 'search_N_cp': 5}
#  {'topN': 5, 'lambda_1': 5, 'lambda_2': 1, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 20, 'search_N_cp': 5}]
# all
# [1535.5, 2778.1176470588234, 2848.157894736842]
# [{'topN': 3, 'lambda_1': 5, 'lambda_2': 1, 'lambda_3': 0.1, 'lambda_4': 10, 'lambda_5': 30, 'beta_1': 10, 'beta_3': 0.1, 'gamma_1': 20, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 5, 'lambda_2': 1, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 30, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 20, 'search_N_cp': 5}
#  {'topN': 5, 'lambda_1': 5, 'lambda_2': 1, 'lambda_3': 0.5, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 20, 'search_N_cp': 5}]

# seg
# [865.5, 1193.111111111111, 1344.0]
# [{'topN': 3, 'lambda_1': 1, 'lambda_4': 10, 'lambda_5': 50, 'beta_1': 20, 'beta_3': 2, 'gamma_1': 30, 'search_N_cp': 3}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 10, 'lambda_5': 40, 'beta_1': 10, 'beta_3': 1, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 1, 'lambda_1': 5, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 15, 'beta_3': 0.5, 'gamma_1': 40, 'search_N_cp': 3}]
# event
# [878.5, 1030.111111111111, 1185.4583333333333]
# [{'topN': 3, 'lambda_1': 1, 'lambda_4': 10, 'lambda_5': 50, 'beta_1': 20, 'beta_3': 2, 'gamma_1': 30, 'search_N_cp': 3}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 10, 'lambda_5': 40, 'beta_1': 10, 'beta_3': 1, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 1, 'lambda_1': 10, 'lambda_4': 10, 'lambda_5': 20, 'beta_1': 15, 'beta_3': 0.5, 'gamma_1': 40, 'search_N_cp': 5}]
# all
# [1744.0, 2223.222222222222, 2562.3333333333335]
# [{'topN': 3, 'lambda_1': 1, 'lambda_4': 10, 'lambda_5': 50, 'beta_1': 20, 'beta_3': 2, 'gamma_1': 30, 'search_N_cp': 3}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 10, 'lambda_5': 40, 'beta_1': 10, 'beta_3': 1, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 1, 'lambda_1': 5, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 15, 'beta_3': 0.5, 'gamma_1': 40, 'search_N_cp': 3}]

# seg
# [605.0, 1290.0833333333333, 1352.8333333333333]
# [{'topN': 3, 'lambda_1': 1, 'lambda_4': 20, 'lambda_5': 20, 'beta_1': 20, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 5, 'lambda_4': 30, 'lambda_5': 50, 'beta_1': 10, 'beta_3': 1, 'gamma_1': 30, 'search_N_cp': 3}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 30, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 40, 'search_N_cp': 3}]
# event
# [452.0, 1025.0, 1144.9166666666667]
# [{'topN': 3, 'lambda_1': 1, 'lambda_4': 20, 'lambda_5': 20, 'beta_1': 20, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 5, 'lambda_4': 20, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 1, 'gamma_1': 40, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 30, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 40, 'search_N_cp': 3}]
# all
# [1057.0, 2462.4166666666665, 2497.75]
# [{'topN': 3, 'lambda_1': 1, 'lambda_4': 20, 'lambda_5': 20, 'beta_1': 20, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 5, 'lambda_4': 30, 'lambda_5': 50, 'beta_1': 10, 'beta_3': 1, 'gamma_1': 30, 'search_N_cp': 3}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 30, 'lambda_5': 20, 'beta_1': 10, 'beta_3': 0.5, 'gamma_1': 40, 'search_N_cp': 3}]

# seg
# [605.0, 720.3636363636364, 775.6666666666666]
# [{'topN': 3, 'lambda_1': 3, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 25, 'lambda_5': 20, 'beta_1': 13, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 4}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 25, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.5, 'gamma_1': 35, 'search_N_cp': 5}]
# event
# [452.0, 920.0, 1008.4285714285714]
# [{'topN': 3, 'lambda_1': 3, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 25, 'lambda_5': 20, 'beta_1': 13, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 4}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 25, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 30, 'search_N_cp': 4}]
# all
# [1057.0, 1640.3636363636363, 2068.037037037037]
# [{'topN': 3, 'lambda_1': 3, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 25, 'lambda_5': 20, 'beta_1': 13, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 4}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 25, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.5, 'gamma_1': 35, 'search_N_cp': 5}]


# def generate_para_bank(args):
#     #  build parameter bank
#     para_bank=[]
#     for topN in [3]:
#         for lambda_1 in [1, 2, 3]:
#             #for lambda_2 in [1, 0.5]:
#                 #for lambda_3 in [0.1, 0.5]:
#                     for lambda_4 in [20, 25]:
#                         for lambda_5 in [20, 30]:
#                                 for beta_1 in [13, 20]:
#                                         for beta_3 in [0.5, 0.7, 1]:
#                                             for gamma_1 in [30, 35]:
#                                                 for search_N_cp in [4, 5]:
#                                                     para_bank.append({'topN':topN, 'lambda_1':lambda_1,
#                                                                       'lambda_4':lambda_4, 'lambda_5':lambda_5,
#                                                                       'beta_1':beta_1, 'beta_3':beta_3, 'gamma_1':gamma_1, 'search_N_cp':search_N_cp})
#
#     random.seed(0)
#     random.shuffle(para_bank)
#     print("Parameter Bank Done! There are {} sets of parameters totally.".format(len(para_bank)))
#     print("="*74)
#     with open(op.join(args.save_path, 'para_bank.p'), 'wb') as f:
#         pickle.dump(para_bank, f)
#
#     return para_bank

# seg
# [805.9333333333333, 805.9333333333333, 813.3333333333334]
# [{'topN': 3, 'lambda_1': 3, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 2, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 3, 'lambda_4': 20, 'lambda_5': 28, 'beta_1': 15, 'beta_3': 0.5, 'gamma_1': 32, 'search_N_cp': 5}]
# event
# [1326.6666666666667, 1326.6666666666667, 1336.2]
# [{'topN': 3, 'lambda_1': 3, 'lambda_4': 20, 'lambda_5': 28, 'beta_1': 17, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 3, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 17, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 1, 'lambda_4': 20, 'lambda_5': 25, 'beta_1': 17, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 5}]
# all
# [2169.4333333333334, 2171.9666666666667, 2171.9666666666667]
# [{'topN': 3, 'lambda_1': 2, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.6, 'gamma_1': 32, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 3, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 2, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 5}]
# seg
# [1326.0, 1652.0, 1818.1153846153845]
# [{'topN': 3, 'lambda_1': 3.5, 'lambda_4': 20, 'lambda_5': 35, 'beta_1': 13, 'beta_3': 0.8, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 2.8, 'lambda_4': 19, 'lambda_5': 33, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 28, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 2.8, 'lambda_4': 19, 'lambda_5': 35, 'beta_1': 12, 'beta_3': 0.8, 'gamma_1': 31, 'search_N_cp': 5}]
# event
# [612.6666666666666, 1110.2941176470588, 1313.0]
# [{'topN': 3, 'lambda_1': 3.5, 'lambda_4': 20, 'lambda_5': 35, 'beta_1': 13, 'beta_3': 0.8, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 2.8, 'lambda_4': 19, 'lambda_5': 33, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 28, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 3, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.8, 'gamma_1': 31, 'search_N_cp': 5}]
# all
# [1938.6666666666667, 2762.294117647059, 3180.423076923077]
# [{'topN': 3, 'lambda_1': 3.5, 'lambda_4': 20, 'lambda_5': 35, 'beta_1': 13, 'beta_3': 0.8, 'gamma_1': 30, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 2.8, 'lambda_4': 19, 'lambda_5': 33, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 28, 'search_N_cp': 5}
#  {'topN': 3, 'lambda_1': 2.8, 'lambda_4': 19, 'lambda_5': 35, 'beta_1': 12, 'beta_3': 0.8, 'gamma_1': 31, 'search_N_cp': 5}]

# 1134
# seg
# [847.9, 1228.5, 1554.2666666666667]
# [{'topN': 3, 'lambda_1': 2.8, 'lambda_4': 19, 'lambda_5': 35, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 28, 'search_N_cp': 5}
#  {'topN': 1, 'lambda_1': 2.8, 'lambda_4': 0, 'lambda_5': -10, 'beta_1': 1, 'beta_3': 0.7, 'gamma_1': 5, 'search_N_cp': 4}
#  {'topN': 5, 'lambda_1': 10, 'lambda_4': 19, 'lambda_5': 1, 'beta_1': 12, 'beta_3': 0.9, 'gamma_1': 30, 'search_N_cp': 6}]
# event
# [550.0, 794.0, 1091.8]
# [{'topN': 1, 'lambda_1': 2.8, 'lambda_4': 0, 'lambda_5': -10, 'beta_1': 1, 'beta_3': 0.7, 'gamma_1': 5, 'search_N_cp': 4}
#  {'topN': 1, 'lambda_1': 10, 'lambda_4': -5, 'lambda_5': 35, 'beta_1': 1, 'beta_3': 0.8, 'gamma_1': 31, 'search_N_cp': 6}
#  {'topN': 8, 'lambda_1': 10, 'lambda_4': 0, 'lambda_5': 35, 'beta_1': 30, 'beta_3': 5, 'gamma_1': 0, 'search_N_cp': 8}]
# all
# [1778.5, 2059.9, 2550.5]
# [{'topN': 1, 'lambda_1': 2.8, 'lambda_4': 0, 'lambda_5': -10, 'beta_1': 1, 'beta_3': 0.7, 'gamma_1': 5, 'search_N_cp': 4}
#  {'topN': 3, 'lambda_1': 2.8, 'lambda_4': 19, 'lambda_5': 35, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 28, 'search_N_cp': 5}
#  {'topN': 1, 'lambda_1': 10, 'lambda_4': -5, 'lambda_5': 35, 'beta_1': 1, 'beta_3': 0.8, 'gamma_1': 31, 'search_N_cp': 6}]


# seg
# [739.0, 1045.4285714285713, 1565.111111111111]
# [{'topN': 3, 'lambda_1': 3, 'lambda_4': 22, 'lambda_5': 25, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 32, 'search_N_cp': 4}
#  {'topN': 3, 'lambda_1': 3, 'lambda_4': 22, 'lambda_5': 25, 'beta_1': 15, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 4}
#  {'topN': 3, 'lambda_1': 3, 'lambda_4': 22, 'lambda_5': 30, 'beta_1': 15, 'beta_3': 0.5, 'gamma_1': 32, 'search_N_cp': 4}]
# event
# [192.0, 781.8571428571429, 1048.5]
# [{'topN': 3, 'lambda_1': 3, 'lambda_4': 22, 'lambda_5': 25, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 32, 'search_N_cp': 4}
#  {'topN': 3, 'lambda_1': 3, 'lambda_4': 22, 'lambda_5': 25, 'beta_1': 15, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 4}
#  {'topN': 3, 'lambda_1': 3, 'lambda_4': 22, 'lambda_5': 30, 'beta_1': 15, 'beta_3': 0.5, 'gamma_1': 32, 'search_N_cp': 4}]
# all
# [931.0, 1827.2857142857142, 2613.6111111111113]
# [{'topN': 3, 'lambda_1': 3, 'lambda_4': 22, 'lambda_5': 25, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 32, 'search_N_cp': 4}
#  {'topN': 3, 'lambda_1': 3, 'lambda_4': 22, 'lambda_5': 25, 'beta_1': 15, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 4}
#  {'topN': 3, 'lambda_1': 3, 'lambda_4': 22, 'lambda_5': 30, 'beta_1': 15, 'beta_3': 0.5, 'gamma_1': 32, 'search_N_cp': 4}]

# [931.0, 2039.0, 2496.0]
# [{'topN': 3, 'lambda_1': 3, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 15, 'beta_3': 0.7, 'gamma_1': 30, 'search_N_cp': 4}
#  {'topN': 3, 'lambda_1': 3, 'lambda_4': 20, 'lambda_5': 30, 'beta_1': 13, 'beta_3': 0.7, 'gamma_1': 30, 'search_N_cp': 4}
#  {'topN': 3, 'lambda_1': 3, 'lambda_4': 20, 'lambda_5': 25, 'beta_1': 13, 'beta_3': 0.5, 'gamma_1': 30, 'search_N_cp': 4}]


def generate_para_bank(args):
    #  build parameter bank
    para_bank=[]
    for topN in [3]:
        for lambda_1 in [1, 2, 3]:
                    for lambda_4 in [20, 22]:
                        for lambda_5 in [25, 28, 30]:
                                for beta_1 in [13, 15, 17]:
                                        for beta_3 in [0.5, 0.6, 0.7]:
                                            for gamma_1 in [30, 32]:
                                                for search_N_cp in [5]:
                                                    para_bank.append({'topN':topN, 'lambda_1':lambda_1,
                                                                      'lambda_4':lambda_4, 'lambda_5':lambda_5,
                                                                      'beta_1':beta_1, 'beta_3':beta_3, 'gamma_1':gamma_1, 'search_N_cp':search_N_cp})

    random.seed(0)
    random.shuffle(para_bank)
    print("Parameter Bank Done! There are {} sets of parameters totally.".format(len(para_bank)))
    print("="*74)
    with open(op.join(args.save_path, 'para_bank.p'), 'wb') as f:
        pickle.dump(para_bank, f)

    return para_bank



def generate_para_bank_P(args):

    para_bank=[]
    for topN in [5]:
            for beta_1 in [15, 1, 0.1]:
                    for beta_3 in [0.5, 1, 10]:
                            for search_N_cp in [5]:
                                    para_bank.append({'topN':topN, 'beta_1':beta_1, 'beta_3':beta_3, 'search_N_cp':search_N_cp})

    random.seed(0)
    random.shuffle(para_bank)
    print("Parameter Bank for P Done! There are {} sets of parameters totally.".format(len(para_bank)))
    print("="*74)
    with open(op.join(args.save_path, 'para_bank_P.p'), 'wb') as f:
        pickle.dump(para_bank, f)
    return para_bank

def generate_para_bank_E(args):

    para_bank=[]
    for topN in [3]:
        for lambda_4 in [20, 22]:
            for lambda_5 in [25, 28, 30]:
                for gamma_1 in [30, 32]:
                    for search_N_cp in [5]:
                        para_bank.append({'topN':topN,
                                          'lambda_4':lambda_4, 'lambda_5':lambda_5,
                                           'gamma_1':gamma_1, 'search_N_cp':search_N_cp})

    random.seed(0)
    random.shuffle(para_bank)
    print("Parameter Bank E Done! There are {} sets of parameters totally.".format(len(para_bank)))
    print("="*74)
    with open(op.join(args.save_path, 'para_bank_E.p'), 'wb') as f:
        pickle.dump(para_bank, f)

    return para_bank
