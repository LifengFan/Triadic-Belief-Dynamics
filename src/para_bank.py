import random
import pickle
import os.path as op

def generate_para_bank(args):

    para_bank=[]
    for topN in [4, 8]:
        for lambda_1 in [0.5, 1, 5]:
            for lambda_2 in [0.5, 1, 5]:
                for lambda_3 in [0.5, 1, 5]:
                
                        for lambda_5 in [0.5, 1, 5]:
                                for lambda_6 in [0.5, 1, 5]:
                                        for lambda_7 in [0.5, 1, 5]:
                                            for lambda_8 in [0.5, 1, 5]:
            
                                                    for search_N_cp in [5, 10]:

                                                        para_bank.append({'topN':topN, 'lambda_1':lambda_1, 'lambda_2':lambda_2, 'lambda_3':lambda_3,
                                                                      'lambda_5':lambda_5, 'lambda_6':lambda_6, 'lambda_7':lambda_7, 'lambda_8':lambda_8,
                                                                       'search_N_cp':search_N_cp})

    random.seed(0)
    random.shuffle(para_bank)
    print("Parameter Bank Done! There are {} sets of parameters totally.".format(len(para_bank)))
    print("="*74)
    with open(op.join(args.save_path, 'para_bank.p'), 'wb') as f:
        pickle.dump(para_bank, f)

    return para_bank


