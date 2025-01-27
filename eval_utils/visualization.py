"""
Modifed based on code from: https://github.com/Y-debug-sys/Diffusion-TS/blob/main/Utils/metric_utils.py
"""

from sklearn.manifold import TSNE
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def TSN_CC(ori_data, fake_data,model):
    size = min([ori_data.shape[0],fake_data.shape[0]])
    print('=======Size========: ',size)
    ori_data = torch.tensor(ori_data)
    fake_data = torch.tensor(fake_data)
    colors = ["red" for i in range(size)] + ["blue" for i in range(size)]
    
    def get_lower_triangular_indices_no_diag(n):
            indices = torch.tril_indices(n, n).long()
            indices_without_diagonal = (indices[0] != indices[1]).nonzero(as_tuple=True)
            return indices[0][indices_without_diagonal], indices[1][indices_without_diagonal]

    index = get_lower_triangular_indices_no_diag(fake_data.shape[2])
    corr_gen = []
    for i in range(size):
        corr_matrix = torch.corrcoef(fake_data[i].T)
        corr_gen.append(corr_matrix[index])
    corr_gen = torch.stack(corr_gen, dim=0)
    corr_gen = torch.where(torch.isinf(corr_gen) | torch.isnan(corr_gen), torch.tensor(0.0), corr_gen)


    corr = []
    for i in range(size):
        corr_matrix = torch.corrcoef(ori_data[i].T)
        corr.append(corr_matrix[index])
    corr = torch.stack(corr, dim=0)
    corr = torch.where(torch.isinf(corr) | torch.isnan(corr), torch.tensor(0.0), corr)
    

    prep_data_final = np.concatenate((corr, corr_gen), axis=0)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(prep_data_final)
    
    f, ax = plt.subplots(1)
    plt.xlim([-15,15])
    plt.ylim([-15,15])
    plt.scatter(tsne_results[:size, 0], tsne_results[:size, 1],
                c=colors[:size], alpha=0.2, label="Original", s = 10,norm ='Normalize')
    plt.scatter(tsne_results[size:, 0], tsne_results[size:, 1],
                c=colors[size:], alpha=0.2, label=model, s = 10,norm ='Normalize')

    ax.legend(loc='upper left',fontsize = 'large',prop={'size': 18},scatterpoints=1,markerscale=3)

    # plt.title('t-SNE plot')
    plt.xlabel('Dimension 1',fontsize=13)
    plt.ylabel('Dimension 2',fontsize=13)
    plt.savefig(f'/sciclone/home/yli102/TS/Figs/{model}-energy-TSN-CC.png')
    
    
def VD_kernel(ori_data, generated_data, model):
    """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - model: model name
    - generated_data: generated synthetic data
    - analysis: tsne or pca or kernel
  """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([generated_data.shape[0], ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]

    # Data preprocessing
    # ori_data = np.asarray(ori_data)
    # generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

       
    # Visualization parameter
    # colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    f, ax = plt.subplots(1)
    sns.distplot(prep_data, hist=False, kde=True, kde_kws={'linewidth': 5}, label='Original', color="red")
    sns.distplot(prep_data_hat, hist=False, kde=True, kde_kws={'linewidth': 5, 'linestyle':'--'}, label=model, color="blue")
    # Plot formatting

    # plt.legend(prop={'size': 22})
    plt.legend(loc='upper left')
    # plt.xlabel('Data Value')
    plt.ylabel('Data Density Estimate')
    plt.xlabel('Value')
    # plt.rcParams['pdf.fonttype'] = 42
    plt.ylim((0, 9))
    # plt.show()
    plt.savefig(f'/Users/LiYang/Desktop/pad/{model}-VD.png')
#     plt.close()



def TSN_VD(ori_data, generated_data, model):
    """Using tSNE for generated and original data values visualization.
  
  Args:
    - ori_data: original data
    - model: model name
    - generated_data: generated synthetic data
  """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([generated_data.shape[0], ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]

    # Data preprocessing
    # ori_data = np.asarray(ori_data)
    # generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    # Do t-SNE Analysis together
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(prep_data_final)

    # Plotting
    f, ax = plt.subplots(1)

    plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                c=colors[:anal_sample_no], alpha=0.2, label="Original")
    plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:], alpha=0.2, label=model)

    ax.legend(loc='upper left')

    # plt.title('t-SNE plot')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(f'/Users/LiYang/Desktop/pad/{model}-TSN-VD.png')

    # plt.show()
