import glob, os, shutil, sys
import numpy as np
import matplotlib.pyplot as plt
import json
import dpdata
from ase import Atoms
from ase.io import write, read
from matplotlib.gridspec import GridSpec
from random import sample
from matplotlib import cm

# conversion unit here, modify if you need
au2eV = 2.72113838565563E+01
au2A = 5.29177208590000E-01


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

def cmap2colorlist(cmap_name, color_numb):
    """gets a list of color from selected matplotlib cmap
    'cmap_name': str, name of cmap, for example 'plasma'.
                 See available cmaps in tutorial "Choosing Colormaps in Matplotlib"
    'color_numb': int, length of desired color list
    Returns to a numpy array. Its shape is (color_numb, 4), 
    and each column (a (4,) vector) represents a color. 
    example:
        clist = cmap2colorlist('plasma', 4)
        for ii in range(4):
            ax.plot(x, y[ii], color=clist[ii])
    """
    colormap = cm.get_cmap(cmap_name, color_numb)
    idx = np.arange(color_numb)
    colorlist = colormap(idx)
    return colorlist

def plot_all_model_devi(target_dir, trust_lo, trust_hi):
    md_list = glob.glob(os.path.join(target_dir, "model_devi.out*"))
    md_list.sort()
    iter_list = []
    task_list = []
    temp_list = []
    for md in md_list:
        md_split = md.split("-")
        iter_list.append(md_split[1])
        task_list.append(md_split[2])
        temp_list.append(md_split[3])
    # remove duplicates
    iter_list = list_uniq(iter_list)
    task_list = list_uniq(task_list)
    temp_list = list_uniq(temp_list)
    # get sys_list
    sys_list = []
    for task in task_list:
        sys_list.append(task.split(".")[1])
    # remove duplicates 
    sys_list = list_uniq(sys_list)
    sys_list.sort()
      
    # get parameter 
    numb_iter = len(iter_list)

    bins = np.linspace(0.00, 1, 400)
    
    for temp in temp_list:
        for _sys in sys_list:
            model_devi_data = []
            for _iter in iter_list:
                md_file_list = glob.glob(os.path.join(target_dir, "*{0}*task.{1}*-{2}*".format(_iter, _sys, temp)))
                one_iter_data = []
                # extract data
                for md_file in md_file_list:
                    one_iter_data.append(np.loadtxt(md_file, usecols=4))
                one_iter_data = np.concatenate(one_iter_data)
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
            print("DPGEN Exploration for System {0} at {1}K".format(_sys, temp))
            print("-"*50)
            idx = 0
            for ac, ca, fa in zip(accurate_ratio, candidate_ratio, failed_ratio):
                print("{0}".format(iter_list[idx]))
                print("accurate : {0:6.2f}%".format(ac))
                print("candidate: {0:6.2f}%".format(ca))
                print("failed   : {0:6.2f}%".format(fa))
                idx += 1

            #plot the figure
            colors = cmap2colorlist('plasma', numb_iter)
            fig = plt.figure(figsize=(14,7))
            fig.suptitle("Active Learning in {0}K".format(temp), fontsize = 25)
            #subplot left
            max_y = []
            ax = fig.add_subplot(1, 2, 1)
            for i in list(range(numb_iter)):
                y, x = np.histogram(model_devi_data[i], bins)
                tot_num = len(model_devi_data[i])
                ax.plot(x[:-1], y/tot_num, alpha=1, label = "iter "+str(i), color=colors[i])
                max_y.append(y.max()/tot_num)
            # text setting
            max_y = np.array(max_y, dtype=float)
            accurate_left = 0.05 
            candidate_left = trust_lo/1.
            fail_left = (trust_hi + 0.2)/1.
            top = 0.95
            ax.set_ylim(0, 2*max_y.max())
            ax.axvline(x=trust_lo, ymin=0, ymax=.8, color='k', ls='--')
            ax.axvline(x=trust_hi, ymin=0, ymax=.8, color='k', ls='--')
            ax.text(accurate_left, top, "Accurate", fontsize=15, 
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes
                   )
            ax.text(candidate_left, top, "Candidate", fontsize=15,
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes
                   )
            ax.text(fail_left, top, "Fail", fontsize=15,
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes
                   )
            ax.legend(prop={'size': 15})
            ax.set_xlabel("max force deviation[eV/A]", fontsize = 20)
            ax.set_ylabel("fraction of frames", fontsize = 20)
            ax.set_title("Histograms of ensemble deviation", fontsize = 20)
            ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
            #subplot right
            ax = fig.add_subplot(1, 2, 2)
            ax.plot(iteration_list, accurate_ratio, alpha=0.6, marker='o', label = "Accurate" )
            ax.plot(iteration_list, candidate_ratio, alpha=0.6, marker='o', label = "Candidate")
            ax.plot(iteration_list, failed_ratio, alpha=0.6, marker='o', label = "Failed" )
        
            ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
            ax.legend(prop={'size': 20})
            ax.set_xlabel("iteration number", fontsize = 20)
            ax.set_ylabel("fraction of frames[%]", fontsize = 20)
            ax.set_title("Fraction Change Over Iteration", fontsize = 20)
            fig.tight_layout()
            fig.savefig(os.path.join(target_dir, "dpgen_model_devi_{0}K_sys{1}".format(temp, _sys)), dpi=400)
         
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

def xyz2npy(pos, atom_num, output, unit_convertion=1.0):
    total = np.empty((0,atom_num*3), float)
    for single_pos in pos:
        tmp=single_pos.get_positions()
        tmp=np.reshape(tmp,(1,atom_num*3))
        total = np.concatenate((total,tmp), axis=0)
    total = total * unit_convertion
    np.save(output, total)

def energy2npy(pos, output, unit_convertion=1.0):
    total = np.empty((0), float)
    for single_pos in pos:
        tmp=single_pos.info.pop('E')
        tmp=np.array(tmp,dtype="float")
        tmp=np.reshape(tmp,1)
        total = np.concatenate((total,tmp), axis=0)
    total = total * unit_convertion
    np.save(output, total)

def cell2npy(pos, output, unit_convertion=1.0):
    total = np.empty((0,9),float)
    frame_num = len(pos)
    cell = pos[0].get_cell()
    cell = cell.reshape(1, 9)
    for frame in range(frame_num):
        total = np.concatenate((total,cell),axis=0)
    total = total * unit_convertion
    np.save(output, total)

def type_raw(single_pos, output, output_2):
    element = single_pos.get_chemical_symbols()
    element = np.array(element)
    tmp, indice = np.unique(element, return_inverse=True)
    np.savetxt(output, indice, fmt='%s',newline=' ')
    np.savetxt(output_2, tmp, fmt='%s')

def cp2k_xyz_to_dpmd(data_path, atom_num, cell):
    # read the pos and frc
    data_path = os.path.abspath(data_path)
    pos_path = os.path.join(data_path, "*pos-1.xyz")
    frc_path = os.path.join(data_path, "*frc-1.xyz")
    print("you are now working on: {0}".format(data_path))
    pos_path = glob.glob(pos_path)[0]
    print("The path of position file is {0}".format(pos_path))
    frc_path = glob.glob(frc_path)[0]
    print("The path of force file is {0}".format(frc_path))
    pos = read(pos_path, index = ":" )
    for i in pos:
        i.set_cell(cell)
        i.set_pbc(True)
    frc = read(frc_path, index = ":" )
    for i in frc:
        i.set_cell(cell)
        i.set_pbc(True)
    # numpy path
    set_path = os.path.join(data_path, "set.000")
    if os.path.isdir(set_path):
        print("detect directory exists\n now remove it")
        shutil.rmtree(set_path)
        os.mkdir(set_path)
    else:
        print("detect directory doesn't exist\n now create it")
        os.mkdir(set_path)
    type_path = os.path.join(data_path, "type.raw")
    type_map_path = os.path.join(data_path, "type_map.raw")
    coord_path = os.path.join(set_path, "coord.npy")
    force_path = os.path.join(set_path, "force.npy")
    box_path = os.path.join(set_path, "box.npy")
    energy_path = os.path.join(set_path, "energy.npy")
    #tranforrmation
    xyz2npy(pos, atom_num, coord_path)
    xyz2npy(frc, atom_num, force_path, au2eV/au2A)
    energy2npy(pos, energy_path, au2eV)
    cell2npy(pos, box_path)
    type_raw(pos[0], type_path, type_map_path)


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

def collect_fp(fp_all_dir, prefix, fin_xyz_prefix):
    """
    example:
        collect_fp(".", "fp.", 'new')
    """
    fp_dir_list = glob.glob(os.path.join(fp_all_dir, "{0}[0-9][0-9][0-9]".format(prefix)))
    fp_dir_list.sort()
    with open("{0}-pos-1.xyz".format(fin_xyz_prefix), "wb") as outfbj:
        for f in fp_dir_list:
            pos_list = glob.glob(os.path.join(f, "*pos-1.xyz"))
            print(pos_list)
            if pos_list:
                with open(pos_list[0], "rb") as infbj:
                    shutil.copyfileobj(infbj, outfbj)
    with open("{0}-frc-1.xyz".format(fin_xyz_prefix), "wb") as outfbj:
        for f in fp_dir_list:
            pos_list = glob.glob(os.path.join(f, "*frc-1.xyz"))
            print(pos_list)
            if pos_list:
                with open(pos_list[0], "rb") as infbj:
                    shutil.copyfileobj(infbj, outfbj)
    with open("{0}-1.ener".format(fin_xyz_prefix), "wb") as outfbj:
        for f in fp_dir_list:
            pos_list = glob.glob(os.path.join(f, "*-1.ener"))
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

def random_ex_data(data_dir, ex_num):
    data=dpdata.LabeledSystem(data_dir, fmt="deepmd/npy")
    new_data = None
    tot_num = len(data)
    random_list = sample(range(tot_num), ex_num)
    for i in random_list:
        if new_data is None:
            new_data = data[i]
        else:
            new_data += data[i]
    return new_data

def random_ex_data_to_dpmd(data_dir, ex_num):
    new_data = random_ex_data(data_dir, ex_num)
    new_data.to_deepmd_npy("new_data")
    print(new_data['cells'][0])

def dpdata_to_poscar_md_init(data, ex_num, output_path):
    new_data = None
    tot_num = len(data)
    random_list = sample(range(tot_num), ex_num)
    for i in random_list:
        if new_data is None:
            new_data = data[i]
        else:
            new_data += data[i]
   
    for idx, i in enumerate(new_data):
        filename = os.path.join(output_path, "POSCAR{0:03d}".format(idx))
        i.to_vasp_poscar(filename)




if __name__ == '__main__':
    #collect_all_lcurves(".", "lcurve_collect")
    #plot_all_lcurves("lcurve_collect", 4)
    collect_all_model_devi(".", "model_devi_collect")
    plot_all_model_devi("model_devi_collect", 0.3, 0.6)
    collect_cp2k_walltime(".", "fp_collect")
    plot_fp_time("fp_collect")

