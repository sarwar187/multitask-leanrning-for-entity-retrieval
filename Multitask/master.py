import os,click,random,string

BASH_SCRIPT_TEMPLATE='''#!/usr/bin/env bash
#SBATCH -J Multitask
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -p titanx-long
'''

EXECUTABLE_PREFIX='python /home/smsarwar/searchie/Multitask/multitask_for_cluster.py '

def generate_script(script_id,paramstring):
    script_name = '/home/smsarwar/searchie/Multitask/scripts/script'+str(script_id)+'.sh'
    script = open(script_name,'w')
    script.write(BASH_SCRIPT_TEMPLATE)
    script.write('#SBATCH -o /home/smsarwar/searchie/Multitask/output/output'+str(script_id) + '.out\n')
    script.write('echo "Starting the execution of task $SLURM_JOBID $SLURM_JOBNAME"\n')
    script.write(EXECUTABLE_PREFIX + paramstring +'\n')
    script.write('echo "Execution ended"' + '\n')
    script.close()
    return script_name

def run_script(script_name):
    command = 'sbatch ' + script_name
    os.system(command)

########################################################################################################################

PARAM_RANGES={'epochs':[200], 'learning_rate':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 'weight_decay':[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,1e-2, 1e-1]}
def main():
    script_id = 1
    for epoch in PARAM_RANGES['epochs']:
        for learning_rate in PARAM_RANGES['learning_rate']:
            for weight_decay in PARAM_RANGES['weight_decay']:
                param_string = str(epoch)+ ' ' + str(learning_rate) + ' ' + str(weight_decay)+' 1 0 /home/smsarwar/searchie/Multitask/wiki.simple.vec /home/smsarwar/searchie/Multitask/model_mincount5.vec' + ' ' + str(script_id)
                script_name = generate_script(script_id, param_string)
                run_script(script_name)
                script_id += 1

main()




