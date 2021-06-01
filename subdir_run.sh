for exp_folder in *
do
	if exp_folder != "subdir_run.sh"
		qsub $exp_folder/job_script
done
