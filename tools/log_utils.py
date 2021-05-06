import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import os; import glob; import IPython
import seaborn as sns

def process_log(log_file, plot_dir):
    print('Processing %s' % log_file)
    # Open and read log file.
    n_lines = n_file_lines(log_file) - 1
    n_data = 0; n_iter = []; loss = []; v_iter = []; v_loss = []
    for i in range(n_lines):
        data_list = read_data_line(log_file, i+1)
        if 'loss:' in data_list:
            if 'val' in data_list:
                val = True
                v_loss.append(float(data_list[-1].split('\n')[0]))
            else:
                val = False
                n_data += 1
                loss.append(float(data_list[-1].split('\n')[0]))
            for k, item in enumerate(data_list):
                if ']' in item:
                    if val: v_iter.append(float(data_list[k].split(']')[0]))
                    else: n_iter.append(float(data_list[k].split(']')[0]))
                    break
    # Save output plots.
    if n_data > 1:
        name = log_file.split('/')[-1].split('.')[0]
        plot_out_dir = '%s/%s/' % (plot_dir, name)
        if not os.path.isdir(plot_out_dir): os.makedirs(plot_out_dir)
        plot_file = plot_out_dir + name
        plot_2D_data([n_iter, loss], ['number of iterations', 'loss'],
                plot_file+ '_loss.png', tikz=True)
        plot_2D_data([v_iter, v_loss], 
                ['number of iterations', 'validation loss'],
                plot_file+ '_val_loss.png', tikz=True)
        ceiling = 2 * np.mean(loss)
        loss = np.array(loss)
        loss[loss > ceiling] = ceiling
        plot_2D_data([n_iter, loss], ['number of iterations', 'loss'],
                plot_file+ '_loss_zoom.png', tikz=True)

def read_data_line(file_name, line_n):
    with open(file_name) as input_file:
        for i, line in enumerate(input_file):
            if i == line_n:
                data_list = line.split(' ')
            elif i > line_n:
                break
    return data_list

def n_file_lines(file_name):
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def print_masks(masks,idx=0,ext=''):
    masks += 1
    masks *= 128
    for i in range(len(masks)):
        cv2.imwrite('./mask/%i_%i_%s.png' %(idx,i,ext), masks[i].cpu().numpy()) 

def plot_vector(vector, ylabel, title, filename):
    plt.ylim((0,1))
    plt.plot(vector, linewidth=4.0)
    plt.xlabel('Frame')
    plt.ylabel(ylabel)
    plt.title(title + ' ('+format(np.min(vector),'.3f')+' min @'+str(np.argmin(vector))+
        ', '+format(np.max(vector),'.3f')+' max @'+str(np.argmax(vector))+')')
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()

def num_unique(video_list):
    unique_list = []
    for _, video in enumerate(video_list):
        if not video[:-1] in unique_list:
            unique_list.append(video[:-1])
    return len(unique_list), unique_list

def read_list_file(file_name):
    video_list = open(file_name,'r').readlines()
    for i in range(len(video_list)):
        video_list[i] = video_list[i].strip('\n')
    return video_list

def bar_plot(idx, labels_all, video_list, name=''):                             
    labels = labels_all[idx]                                                    
    n_frames = len(labels)                                                      
    x = np.arange(n_frames)                                                     
    plt.bar(x, labels)                                                          
    plt.savefig(('./plots/%s%s_bar.png' % (name,video_list[idx])),              
        bbox_inches='tight')                                                    
    tikz_save(('./plots/%s%s_bar.tex' % (name,video_list[idx])))                
    plt.clf()                                                                   

def bar_plot_simple(labels, name=''):
    x = np.arange(len(labels))
    plt.bar(x, labels)
    plt.savefig(('./plots/%s_bar.png' % (name)),                
        bbox_inches='tight')                                                    
    tikz_save(('./plots/%s_bar.tex' % name))                
    plt.clf()

def bar_plot_highlight(labels, h_idx, name=''):
    x = np.arange(len(labels))
    plt.bar(x, labels)
    plt.bar(h_idx, labels[h_idx])
    plt.title(name)
    plt.savefig(('./plots/%s_bar.png' % (name)),                
        bbox_inches='tight')                                                    
    tikz_save(('./plots/%s_bar.tex' % name))                
    plt.clf()                                                               
                                                                                
def bar_plot_compare(idx, labels_all, video_list, rank_bn_all, name=''):                
    labels = labels_all[idx]                                                    
    labels_bs = labels[rank_bn_all[idx]]                                        
    n_frames = len(labels)                                                      
    x = np.arange(n_frames)                                                     
    # GT plot
    plt.bar(x, labels)                                 
    plt.savefig(('./plots/%s%s_bar_gt.png' % (name,video_list[idx])),              
        bbox_inches='tight')                                                    
    tikz_save(('./plots/%s%s_bar_gt.tex' % (name,video_list[idx])))                
    plt.clf()  
    # BN plot
    plt.bar(x, labels_bs)                            
    plt.savefig(('./plots/%s%s_bar.png' % (name,video_list[idx])),              
        bbox_inches='tight')                                                    
    tikz_save(('./plots/%s%s_bar.tex' % (name,video_list[idx])))                
    plt.clf()  
    if 0:
        labels.sort()
        plt.bar(x, labels)                                 
        plt.savefig(('./plots/%s%s_bar_gt_sorted.png' % (name,video_list[idx])),              
            bbox_inches='tight')                                                    
        tikz_save(('./plots/%s%s_bar_gt_sorted.tex' % (name,video_list[idx])))                
        plt.clf()  

def bar_plot_both(idx, labels_all, video_list, rank_bn_all, name=''):                
    labels = labels_all[idx]                                                    
    labels_bs = labels[rank_bn_all[idx]]                                        
    n_frames = len(labels)                                                      
    x = np.arange(n_frames)                                                     
    w = 0.5                                                                     
    plt.bar(x, labels, width=w, align='center')                                 
    plt.bar(x+w, labels_bs, width=w, align='center')                            
    plt.savefig(('./plots/%s%s_bar.png' % (name,video_list[idx])),              
        bbox_inches='tight')                                                    
    tikz_save(('./plots/%s%s_bar.tex' % (name,video_list[idx])))                
    plt.clf()  

def hist_vectors(vectors, name='', labels='', n_bins = 35): 
    bins = np.linspace(0,1,n_bins)
    if 1:
        for i, vector in enumerate(vectors):
            if labels=='':
                sns.distplot(vector, bins=bins, label=str(i))
            else:
                plt.hist(vector, bins, alpha=0.5, label=labels[i])  
                #sns.distplot(vector, bins=bins, label=labels[i])
    else:
            plt.hist(vectors, bins, alpha=0.5, label=labels)    
    plt.legend(loc='upper right')
    plt.savefig(('./plots/%s_%ibins_hist.png' % (name,n_bins)),             
        bbox_inches='tight')                                                    
    tikz_save(('./plots/%s_%ibins_hist.tex' % (name,n_bins)))
    plt.clf()  

def pdf_vectors(vectors, name='', labels='', bw = 0.2): 
    for i, vector in enumerate(vectors):
        if labels=='':
            sns.distplot(vector, bins=bins, label=str(i))
        else:
            sns.kdeplot(vector, bw=bw, label=labels[i], shade=True)
            #sns.distplot(vector, bins=bins, label=labels[i])
    plt.legend(loc='upper right')
    plt.savefig(('./plots/%s_%ibw_pdf.png' % (name, int(bw*100))),              
        bbox_inches='tight')                                                    
    tikz_save(('./plots/%s_%ibw_pdf.tex' % (name, int(bw*100))))
    plt.clf()  

def plot_2D_data(vectors, labels, filename, use_scatter = False, tikz = False):
    if use_scatter:
        plt.scatter(vectors[0], vectors[1])
    else:
        plt.plot(vectors[0], vectors[1])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    plt.grid()
    plt.savefig(filename, bbox_inches='tight')
    if tikz:
        tikz_save(filename.split('.png')[0] + '.tex')
    plt.clf()

def build_vector_from_idx(idx_vector, matrix):
    n_data = len(idx_vector)
    vector = np.zeros(n_data)
    for i, idx in enumerate(idx_vector):
        vector[i] = matrix[i][idx[-1]]
    return vector

def print_out_text(file_name, text, arg='a'):
    output_file = open(file_name, arg)
    output_file.write(str(text))
    output_file.close()

def print_statements(file_name, statements):
    for _, statement in enumerate(statements):
        print(statement)
    output_file = open(file_name, 'a')
    for _, statement in enumerate(statements):
        output_file.write(str(statement))
    output_file.close()

def print_stats_prev(file_name, vector, vector_name, scale=1, dec=1):
    statements = []
    vector *= scale
    mean = np.mean(vector)
    min_ = min(vector)
    max_ = max(vector)
    med = np.median(vector)
    std = np.std(vector)
    cv = std/mean
    statements.append('\nStats for %s:\n' % vector_name)
    statements.append('Mean: %0.1f, Med.: %0.1f, Min.: %0.1f, Max.: %0.1f, Std.Dev.: %0.1f, CofVar.: %0.2f\n' 
        % (mean, med, min_, max_, std, cv))
    statements.append('%s & %0.1f & %0.1f & %0.1f & %0.1f & %0.2f\n' 
        % (vector_name, mean, med, min_, max_, cv))
    '''
    statements.append('Mean: %0.3f, Med.: %0.3f, Min.: %0.3f, Max.: %0.3f, Std.Dev.: %0.3f, CofVar.: %0.3f' 
        % (mean, med, min_, max_, std, cv))
    statements.append('%s & %0.3f & %0.3f & %0.3f & %0.3f & %0.3f' 
        % (vector_name, mean, med, min_, max_, std))
    '''
    print_statements(file_name,statements)

def print_stats(file_name, vector, vector_name, scale=1, dec=1):
    statements = []
    vector *= scale
    mean = np.mean(vector)
    min_ = min(vector)
    max_ = max(vector)
    med = np.median(vector)
    std = np.std(vector)
    cv = std/mean
    statements.append('\nStats for %s:\n' % vector_name)
    statements.append('Mean: %0.1f, Med.: %0.1f, Min.: %0.1f, Max.: %0.1f, Std.Dev.: %0.1f, CofVar.: %0.2f\n' 
        % (mean, med, min_, max_, std, cv))
    statements.append('%s & %0.1f & %0.1f & %0.1f--%0.1f & %0.2f\n' 
        % (vector_name, mean, med, min_, max_, cv))
    '''
    statements.append('Mean: %0.3f, Med.: %0.3f, Min.: %0.3f, Max.: %0.3f, Std.Dev.: %0.3f, CofVar.: %0.3f' 
        % (mean, med, min_, max_, std, cv))
    statements.append('%s & %0.3f & %0.3f & %0.3f & %0.3f & %0.3f' 
        % (vector_name, mean, med, min_, max_, std))
    '''
    print_statements(file_name,statements)
