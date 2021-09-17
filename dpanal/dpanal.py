import glob, os, shutil
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.gridspec import GridSpec


def list_uniq(list_var):
    list_var = set(list_var)
    list_var = list(list_var)
    list_var.sort()
    return list_var
   

def create_path(path):
    path += '/'
    if os.path.isdir(path): 
        dirname = os.path.dirname(path)        
        counter = 0
        while True :
            bk_dirname = dirname + ".bk%03d" % counter
            if not os.path.isdir(bk_dirname): 
                shutil.move (dirname, bk_dirname) 
                break
            counter += 1
    os.makedirs (path)

def collect_all_lcurves(dpgen_dir, target_dir):
    create_path(target_dir)
    iter_dir_list = glob.glob(os.path.join(dpgen_dir, "iter.*"))
    iter_dir_list.sort()
    for iter_dir in iter_dir_list:
        iter_bn = os.path.basename(iter_dir)
        train_dir_list = glob.glob(os.path.join(iter_dir, "00.train", "0*"))
        train_dir_list.sort()
        for train_dir in train_dir_list:
            train_bn = os.path.basename(train_dir)
            print("Now work on {0}: {1}".format(iter_bn, train_bn))
            src = os.path.join(train_dir, "lcurve.out")
            dst = os.path.join(target_dir, "lcurve.out-{0}-{1}".format(iter_bn, train_bn))
            shutil.copy(src, dst) 
           
def plot_all_lcurves(target_dir, numb_model=4):
    lcurve_list = glob.glob(os.path.join(target_dir, "*iter.*"))
    lcurve_list.sort()
    numb_iter = int(len(lcurve_list)/numb_model)

    fig = plt.figure(figsize=(4*numb_model,4*numb_iter), dpi=200)
    gs = GridSpec(numb_iter, numb_model, figure=fig) 

    for iter_idx in range(numb_iter):
        for model_idx in range(numb_model):
            lcurve_file = lcurve_list[numb_model*iter_idx+model_idx]
            step = np.loadtxt(lcurve_file, usecols = 0)
            trn_ener = np.loadtxt(lcurve_file, usecols = 4)
            tst_ener = np.loadtxt(lcurve_file, usecols = 3)

            ax = fig.add_subplot(gs[numb_model*iter_idx+model_idx])
            ax.scatter(step, trn_ener, s=10, label="trn-ener")
            ax.scatter(step, tst_ener, s=10, label="tst-ener")
            ax.set_title('iteration{0}: model{1}'.format(iter_idx, model_idx))
            ax.legend()
            ax.set_yscale('log')
            lcurve_bn = os.path.basename(lcurve_file)
            print(lcurve_bn)
        
    gs.tight_layout(fig)
    fig.savefig(os.path.join(target_dir, "all-train-energy.png"))

    fig = plt.figure(figsize=(4*numb_model,4*numb_iter), dpi=200)
    gs = GridSpec(numb_iter, numb_model, figure=fig) 

    for iter_idx in range(numb_iter):
        for model_idx in range(numb_model):
            lcurve_file = lcurve_list[numb_model*iter_idx+model_idx]
            step = np.loadtxt(lcurve_file, usecols = 0)
            trn_frc = np.loadtxt(lcurve_file, usecols = 6)
            tst_frc = np.loadtxt(lcurve_file, usecols = 5)

            ax = fig.add_subplot(gs[numb_model*iter_idx+model_idx])
            ax.scatter(step, trn_frc, s=10, label="trn-force")
            ax.scatter(step, tst_frc, s=10, label="tst-force")
            ax.set_title('iteration{0}: model{1}'.format(iter_idx, model_idx))
            ax.legend()
            ax.set_yscale('log')
            lcurve_bn = os.path.basename(lcurve_file)
            print(lcurve_bn)
        
    gs.tight_layout(fig)
    fig.savefig(os.path.join(target_dir, "all-train-force.png"))

def collect_all_model_devi(dpgen_dir, target_dir):
    create_path(target_dir)
    iter_dir_list = glob.glob(os.path.join(dpgen_dir, "iter.*"))
    iter_dir_list.sort()
    for iter_dir in iter_dir_list:
        iter_bn = os.path.basename(iter_dir)
        md_dir_list = glob.glob(os.path.join(iter_dir, "01.model_devi", "task.0*"))
        md_dir_list.sort()
        for md_dir in md_dir_list:
            md_bn = os.path.basename(md_dir)
            print("Now work on {0}: {1}".format(iter_bn, md_bn))
            with open (os.path.join(md_dir, "job.json")) as f:
                data=json.load(f)
            src = os.path.join(md_dir, "model_devi.out")
            dst = os.path.join(target_dir, "model_devi.out-{0}-{1}-{2}-{3}".format(iter_bn, md_bn, data['temps'], data['press']))
            shutil.copy(src, dst) 

def plot_all_model_devi(target_dir, trust_lo, trust_hi):
    md_list = glob.glob(os.path.join(target_dir, "model_devi.out*"))
    md_list.sort()
    iter_list = []
    temp_list = []
    for md in md_list:
        md_split = md.split("-")
        iter_list.append(md_split[1])
        temp_list.append(md_split[3])
    # remove duplicates
    iter_list = list_uniq(iter_list)
    temp_list = list_uniq(temp_list)
    
    # get parameter 
    numb_iter = len(iter_list)

    bins = np.linspace(0.00, 1, 400)
    
    for temp in temp_list:
        model_devi_data = []
        for _iter in iter_list:
            md_file_list = glob.glob(os.path.join(target_dir, "*{0}*{1}*".format(_iter, temp)))
            one_iter_data = []
            # extract data
            for md_file in md_file_list:
                one_iter_data.append(np.loadtxt(md_file, usecols=4))
            one_iter_data = np.array(one_iter_data)
            one_iter_data = one_iter_data.flatten()
            model_devi_data.append(one_iter_data)
        iteration_list = []
        for i in range(numb_iter):
            iteration_list.append(str(i))
        #calculate ratio
        accurate_ratio = []
        for i in range(numb_iter):
            tmp = model_devi_data[i]
            ratio = len(tmp[tmp<trust_lo])/len(tmp)
            accurate_ratio.append(ratio*100)
        candidate_ratio = []
        for i in range(numb_iter):
            tmp = model_devi_data[i]
            ratio = len(tmp[np.logical_and( tmp>=trust_lo, tmp <=trust_hi)])/len(tmp)
            candidate_ratio.append(ratio*100)
        failed_ratio = []
        for i in range(numb_iter):
            tmp = model_devi_data[i]
            ratio = len(tmp[tmp>trust_hi])/len(tmp)
            failed_ratio.append(ratio*100)
        #plot the figure
        plt.figure(figsize=(14,7))
        plt.suptitle("Active Learning in {0}K".format(temp), fontsize = 25)
        #subplot left
        plt.subplot(1, 2, 1)
        for i in list(range(numb_iter)):
            y, x = np.histogram(model_devi_data[i], bins)
            tot_num = len(model_devi_data[i])
            plt.plot(x[:-1], y/tot_num, alpha=1, label = "iter "+str(i))
        # text setting
        plt.vlines(trust_lo, 0, 0.2)
        plt.vlines(trust_hi, 0, 0.2)
        plt.text(0.00, 0.16, "Accurate", fontsize=15)
        plt.text(trust_lo, 0.16, "Candidate", fontsize=15)
        plt.text(trust_hi, 0.16, "Fail", fontsize=15)
        plt.legend(prop={'size': 15})
        plt.xlabel("max force deviation[eV/A]", fontsize = 20)
        plt.ylabel("fraction of frames", fontsize = 20)
        plt.title("Histograms of ensemble deviation", fontsize = 20)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        #subplot right
        plt.subplot(1, 2, 2)
        plt.plot(iteration_list, accurate_ratio, alpha=0.6, marker='o', label = "Accurate" )
        plt.plot(iteration_list, candidate_ratio, alpha=0.6, marker='o', label = "Candidate"  )
        plt.plot(iteration_list, failed_ratio, alpha=0.6, marker='o', label = "Failed" )
    
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.legend(prop={'size': 20})
        plt.xlabel("iteration number", fontsize = 20)
        plt.ylabel("fraction of frames[%]", fontsize = 20)
        plt.title("Fraction Change Over Iteration", fontsize = 20)
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, "dpgen_model_devi_{0}K".format(temp)), dpi=400)
         
    # gether data according to temperature
   # md_name_list = np.array(md_name_list)
   # md_name_list.sort(axis=0)
   # print(md_name_list)
def to_xyz(dpmd_data_dir):
    """
    example:
    dpmd_data_dir = "tio2110"
    to_xyz(dpmd_data_dir) 
    """
    data_set = dpdata.LabeledSystem(dpmd_data_dir, fmt="deepmd/npy")

    atom_types = data_set['atom_types']
    atom_names = data_set['atom_names']

    atom_names = np.array(atom_names)
    chemical_symbols = atom_names[atom_types]

    poses = []

    for i in range(len(data_set)):
        pos = Atoms(chemical_symbols)
      #  pos.set_chemical_symbols(chemical_symbols)
        pos.set_positions(data_set['coords'][i])
        poses.append(pos)

    write("reftraj.xyz", poses)


def multiple_reftraj(reftraj_file, stride, other_files):
    """
    example:
    reftraj_file = "reftraj.xyz"
    stride = 50
    other_files = ['cp2k.lsf', 'input.inp', 'rutile.xyz']
    multiple_reftraj(reftraj_file, stride, other_files)
    """
    poses = read(reftraj_file, index = ":")
    num_fp_dir = int(len(poses)/stride) + 1
    for i in range(num_fp_dir):
        fp_dir = 'fp.{0:03d}'.format(i)
        os.mkdir(fp_dir)
        write(os.path.join(fp_dir, 'reftraj.xyz'), poses[stride*i:stride*(i+1)])
        for onefile in other_files:
            src = onefile
            dst = os.path.join(fp_dir, src)
            shutil.copyfile(src, dst)

def collect_fp(fp_all_dir, prefix, fin_xyz_filename):
    """
    example:
        collect_fp(".", "fp.", 'new.xyz')
    """
    fp_dir_list = glob.glob(os.path.join(fp_all_dir, "{0}[0-9][0-9][0-9]".format(prefix)))
    fp_dir_list.sort()
    with open(fin_xyz_filename, "wb") as outfbj:
        for f in fp_dir_list:
            pos_list = glob.glob(os.path.join(f, "*pos-1.xyz"))
            print(pos_list)
            if pos_list:
                with open(pos_list[0], "rb") as infbj:
                    shutil.copyfileobj(infbj, outfbj)

def get_cpk_converge_and_walltime(cp2k_output_file):
    f = open(cp2k_output_file, "r")
    is_converge = True
    while True:
        content = f.readline()
        if 'SCF run NOT converged' in content:
            is_converge = False
        if 'T I M I N G' in content:
            for jj in range(5):
                content = f.readline()
            walltime = content.split()[-1]
            if is_converge:
                return walltime, '1'
            else:
                return walltime, '0'
            break

def collect_cp2k_walltime(dpgen_dir, target_dir):
    """ collect cp2k fp wall time and convergence
    usage: 
        collect_cp2k_wall_time(".", "fp_collect")
        will make dir 'fp_collect';
        write file 'fp_collect/cp2k_walltime-iter*' for each iteration, 
        data format (walltime, convergence) ~ (float, 0/1)
    """
    create_path(target_dir)
    iter_dir_list = glob.glob(os.path.join(dpgen_dir, "iter.*"))
    iter_dir_list.sort()
    iter_dir_list = iter_dir_list[0:-1]
    for iter_dir in iter_dir_list:
        iter_bn = os.path.basename(iter_dir)
        print("Now working on {:s}".format(iter_bn))
        fp_output_dir_list = glob.glob(os.path.join(iter_dir, "02.fp", "task.0*/output"))
        fp_output_dir_list.sort()
        output = open(os.path.join(target_dir, "cp2k_walltime-{:s}".format(iter_bn)), 'w')
        for fp_output in fp_output_dir_list:
            time, convergence = get_cpk_converge_and_walltime(fp_output)
            output.writelines("{0:s}\t{1:s}".format(time, convergence)+"\n")
        output.close()
            

def plot_fp_time(target_dir):
    fp_list = glob.glob(os.path.join(target_dir, "*iter.*"))
    fp_list.sort()
    numb_fp = len(fp_list)
    numb_iter = [os.path.basename(jj) for jj in fp_list]
    numb_iter = [int(jj.split('.')[-1]) for jj in numb_iter]

    fig = plt.figure(figsize=(4, 4*numb_fp))
    gs = GridSpec(numb_fp, 1, figure=fig) 

    for iter_idx in range(numb_fp):
        fptime_file = fp_list[iter_idx]
        fptime = np.loadtxt(fptime_file, usecols = 0)
        fptime = fptime/60
        task = np.arange(len(fptime))
        mean = fptime.mean()
        isconv = np.loadtxt(fptime_file, usecols = 1)
        conv = (isconv == 1)
        not_conv = (isconv == 0)
        tmp = fptime * isconv
        mean_exclude = tmp.mean()
        print("now working on iter-{:06d}".format(numb_iter[iter_idx]))
        # plot each fp
        ax = fig.add_subplot(gs[iter_idx])
        ax.scatter(task[conv], fptime[conv], color = 'b',label="converged")
        ax.scatter(task[not_conv], fptime[not_conv], color = 'r',label="not converged")
        ax.axhline(y=mean, color='k', label="mean cal time")
        ax.set_title('iteration{0}, mean cal time is {1:.2f}'.format(numb_iter[iter_idx], mean))
        ax.set_xlabel("task number")
        ax.set_ylabel("CP2K wall time [min]")
        ax.legend()

    gs.tight_layout(fig)
    fig.savefig(os.path.join(target_dir, "all-fptime.png"), format="PNG", dpi=300)

if __name__ == '__main__':
    #collect_all_lcurves(".", "lcurve_collect")
    #plot_all_lcurves("lcurve_collect", 4)
    collect_all_model_devi(".", "model_devi_collect")
    plot_all_model_devi("model_devi_collect", 0.3, 0.6)
    collect_cp2k_walltime(".", "fp_collect")
    plot_fp_time("fp_collect")

