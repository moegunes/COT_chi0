#!/bin/bash
#
# =====================================================================================	
# SLURM Job Submission Script with Energy Consumption Tracking				
# =====================================================================================	

# === SLURM Directives ================================================================ 
#											
#SBATCH --job-name=connector               # Job name						
#SBATCH --output=job_%j.out         # Standard output file (%j expands to jobID)	
#SBATCH --error=job_%j.err          # Standard error file (%j expands to jobID) 	
#SBATCH --ntasks=1               # Number of tasks (processes)               	
#SBATCH --cpus-per-task=48          # Number of CPU cores per task              	
#SBATCH --partition=TRIXS         # Partition/queue name                      	
#SBATCH --nodes=1                   # Number of nodes                           	
                                                                               	
# === Optional SLURM Directives (uncomment as needed) ===                       	
###SBATCH --time=01:00:00             # Maximum runtime (HH:MM:SS)              	
###SBATCH --mem=8G                    # Total memory required                   	
###SBATCH --account=my_account        # Account/project to charge			
###SBATCH --mail-type=ALL             # Mail events (BEGIN, END, FAIL, ALL)		
###SBATCH --mail-user=user@example.com # Email address for notifications		
###SBATCH --dependency=afterok:12345 # Job dependency					
###SBATCH --array=1-10               # Job array					
											
# ===================================================================================== 


# === Environment Setup =============================================================== 
#											
echo "========== Job Information =========="						
echo "Job ID: $SLURM_JOB_ID"								
echo "Running on host: $(hostname)"							
echo "Start time: $(date)"								
echo "Working directory: $(pwd)"							
echo "====================================="						
#											
# ===================================================================================== 


# === Load necessary modules ========================================================== 
#											
source /etc/profile.d/modules.sh	
module purge                     	# Clear all previously loaded modules		
module load full-package/intel-2017
module load python/3.11.2/hpc-env							
#module load abinit/8.10.2/intel-2025
#module load abinit/9.10.3/intel-2017 
module list										
#											
# ===================================================================================== 

# === Record initial energy consumption =============================================== 
#                                                                                       
ENERGY_START=$(sacct -j $SLURM_JOB_ID --format=ConsumedEnergy --noheader 2>/dev/null)   
if [ -z "$ENERGY_START" ]; then                                                         
    echo "Energy accounting not available or job too new"                               
    ENERGY_START=0									
fi											
#											
# ===================================================================================== 


# === Application Execution =========================================================== 
#                                                                                       
# Record start time for performance metrics						
START_TIME=$(date +%s.%N)								
 											
# Set variables for OpenMP (comment if not applicable)							
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK						
export OMP_STACKSIZE=1G


											
# Example: Run an MPI application							
echo "Starting application at $(date)"							
											
EXE=abinit
source .venv/bin/activate
#srun --cpu-bind=cores python run_connector_isol.py --filenames &> cot_isol.out 			
srun --cpu-bind=cores python3 run.py &> cot.out			
											
# Example: For a Python script								
# python3 my_script.py --input input.dat --output output/result.out		
											
# Record end time for performance metrics						
END_TIME=$(date +%s.%N)									
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc)					
#											
# ===================================================================================== 


# === Record final energy consumption ================================================= 
#                                                                                       
sleep 10  # Wait for energy accounting to update					
ENERGY_END=$(sacct -j $SLURM_JOB_ID --format=ConsumedEnergy --noheader 2>/dev/null)	
if [ -z "$ENERGY_END" ]; then								
    echo "Energy accounting not available"						
    ENERGY_USED="Unknown"								
else											
    ENERGY_USED=$ENERGY_END								
fi

prefix=cot
finp=$prefix.json
fout=$prefix.out
ferr=$prefix.err											
#											
# ===================================================================================== 

# === Generate Report ================================================================= 
#                                                                                       
echo "========== Job Summary =========="						
echo "Job ID: $SLURM_JOB_ID"								
echo "Job Name: $SLURM_JOB_NAME"							
echo "Node Count: $SLURM_JOB_NUM_NODES"							
echo "CPU Count: $SLURM_NTASKS"								
echo "Execution time: $ELAPSED_TIME seconds"						
echo "Energy consumed: $ENERGY_USED"							
echo "End time: $(date)"								
echo "================================="						
#											
# ===================================================================================== 

# === Detailed Energy Report ========================================================== 
#                                                                                       
echo "========== Detailed Energy Report =========="					
sacct -j $SLURM_JOB_ID --format=JobID,JobName,NNodes,NCPUs,NTasks,Elapsed,ConsumedEnergy,ReqMem

# Calculate energy efficiency								
echo "Energy efficiency analysis:"							
if [ "$ENERGY_USED" != "Unknown" ]; then						
    # Calculate energy per node-hour							
    NODE_HOURS=$(echo "scale=6; $ELAPSED_TIME * $SLURM_JOB_NUM_NODES / 3600" | bc)	
    ENERGY_PER_NODE_HOUR=$(echo "scale=6; $ENERGY_USED / $NODE_HOURS" | bc 2>/dev/null)	
    echo "Energy per node-hour: $ENERGY_PER_NODE_HOUR joules"				
   
    # Calculate energy per task								
    ENERGY_PER_TASK=$(echo "scale=6; $ENERGY_USED / $SLURM_NTASKS" | bc 2>/dev/null)	
    echo "Energy per task: $ENERGY_PER_TASK joules"					
    											
    # Convert to kWh for more readable units						
    KWH=$(echo "scale=6; $ENERGY_USED / 3600000" | bc 2>/dev/null)			
    echo "Total energy consumed: $KWH kWh"						
fi											
echo "==========================================="					
#											
# ===================================================================================== 

# === Clean up (optional) ============================================================= 
#											
# Remove temporary files
# rm -f ./tmp*
#											
# ===================================================================================== 

exit 0

