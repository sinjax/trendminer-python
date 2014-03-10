def prepare_folds(ndays, nfolds, step=5, t_size = 48, v_size=8):
	set_fold = [];
	for i in range(nfolds):
		total = i * step + t_size;
		training = range(total - v_size)
		test = range(step);
		validation = range(v_size);
		j = 0;	
		traini = 0;
		tt = round(total/2.)-1;
		while j < tt - v_size/2:
			training[traini] = j;
			j+=1
			traini+=1
		k=0
		while k < len(validation):
			validation[k] = j;
			k+=1
			j+=1

		while j < total:
			training[traini] = j;
			j+=1
			traini+=1
		k=0
		while k < len(test):
			test[k] = j;
			k+=1
			j+=1
		foldi = {
			"training":training,
			"test": test,
			"validation":validation
		}
		set_fold += [foldi]
		return set_fold;