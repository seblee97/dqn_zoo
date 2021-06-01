for exp_folder in $ls
do
	qsub $exp_folder/job_script
done
