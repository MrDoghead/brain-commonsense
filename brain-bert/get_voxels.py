import numpy as np
import json

def get_top_voxels(corrs,top_k=10):
    #print(corrs)
    #print(corrs.shape)

    avg_corrs = np.mean(corrs,0)
    top_k_voxel_ind = np.argsort(avg_corrs)[::-1][0:top_k]
    #print(top_k_voxel_ind)
    return top_k_voxel_ind

if __name__=='__main__':
    layer_voxel = {}
    k = 100
    for i in range(1,13):
        print('loading layer',i)
        res = np.load('./output_bert/predict_F_with_bert_layer_{}_len_10.npy'.format(i), allow_pickle=True)
        corrs = res.item()['corrs_t']
        voxels = get_top_voxels(corrs,k)
        print(voxels.tolist())
        layer_voxel['layer_{}'.format(i)] = voxels.tolist()
    #print(layer_voxel)
    #np.save('bert_voxel_{}.npy'.format(k),layer_voxel)
    with open('bert_voxel_{}.json'.format(k),'w') as f:
        json.dump(layer_voxel,f)

