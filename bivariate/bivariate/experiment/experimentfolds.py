from IPython import embed

def prepare_folds(args):
	ndays = args.ndays
	nfolds = args.nfolds
	t_size=args.ntrain
	step=args.ntest
	v_size = args.nval
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

def prepare_folds_windowed(args):
	ndays = args.ndays
	nfolds = args.nfolds
	training_window=args.train_win
	test_window=args.test_win
	set_fold = []
	for i in range(nfolds):
		validation = []
		train_start = i * test_window
		train_end = train_start + training_window
		training = [x for x in range(train_start, train_end)]
		test = [x for x in range(train_end, train_end + test_window)]

		set_fold += [{
			"training":training,
			"test": test,
			"validation": validation
		}]

	return set_fold
import experimentinputmode as eim

def prepare_fold_args(parser):
	subparsers = parser.add_subparsers()
	
	parser_windowed = subparsers.add_parser('windowed')
	parser_windowed.add_argument('--train-win', type=int, default=48)
	parser_windowed.add_argument('--test-win', type=int, default=5)
	parser_windowed.set_defaults(folds=prepare_folds_windowed)
	eim.prepare_input_mode(parser_windowed)

	parser_growing = subparsers.add_parser('growing')
	parser_growing.add_argument("--n-training", dest="ntrain",default=48,
							help="Number of training instances in the first fold")
	parser_growing.add_argument("--n-validation", dest="nval",default=8,
							help="Number of training instances in the first fold to be used as validation")
	parser_growing.add_argument("--n-test", dest="ntest",default=5,
							help="Number of test instances and the increment per fold")
	parser_growing.set_defaults(folds=prepare_folds)
	eim.prepare_input_mode(parser_growing)