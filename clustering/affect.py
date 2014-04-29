function [clu,W_bar,alpha_est,eigenvectors,eigenvalues] ...
	= batch_affect_spectral(ids,W,num_clust,varargin)
% batch_affect_spectral(ids,W,num_clust) performs AFFECT evolutionary
% spectral clustering with adaptively estimated forgetting factor.
% 
% ids is a length T cell array corresponding to the object IDs of the rows
% and columns of the similarity matrices specified by the length T cell
% array W. The object IDs at each time step must be stored in a cell array
% of strings. The similarity matrices can be either full or sparse matrices.
% 
% num_clust specifies the number of clusters and can either be a scalar, 
% vector of length T, or a string specifying the name of a cluster selection
% heuristic to apply. The choices for heuristic are
%	- 'modularity': calculate the first m eigenvectors, perform
%	  spectral clustering with 2 to m clusters, and select the number
%	  of clusters with maximum modularity.
%	- 'silhouette': similar to modularity except that the number of
%	  clusters with the maximum average silhouette index is chosen.
%	- 'eigengap': plot the first m eigenvalues of the Laplacian matrix
%	  obtained from the mean similarity matrix over all time steps and ask
%	  the user to select the number of clusters. This method results in a
%	  fixed number of clusters over all time steps.
%	- 'eigengap_var': plot the first m eigenvalues of the Laplacian matrix
%	  obtained from each similarity matrix. This method allows the number
%	  of clusters to vary over time, but requires the user to select the
%	  number of clusters at each iteration during each time step.
% m is specified through the optional parameter max_clust, described below.
% 
% Additional parameters are specified in ('name',value) pairs, e.g.
% batch_affect_spectral(ids,W,num_clust,'name1',value1,'name2',value2) and
% are as follows:
%	- max_clust (default: 10): Maximum number of clusters for cluster
%	  selection heuristics. Has no effect when number of clusters is
%	  specified by the user.
%	- alpha (default: 'estimate'): The choice of forgetting factor alpha
%	  can be a scalar between 0 and 1 (for constant forgetting factor) or 
%	  the string 'estimate' to adaptively estimate the forgetting factor at
%	  each time step.
%	- num_iter (default: 3): Number of iterations to use when estimating
%	  forgetting factor. Has no effect for constant forgetting factor.
%	- initialize (default: 'previous'): How to initialize iterative
%	  estimation of alpha. Choices are to initialize with the previous
%	  clusters ('previous') or to first perform clustering with alpha = 0
%	  ('ordinary'). Has no effect for constant forgetting factor.
%	- objective (default: 'NC'): The spectral clustering objective function
%	  to optimize. Choices are average association ('AA'), ratio cut
%	  ('RC'), and normalized cut ('NC').
%	- eig_type (default: 'lanczos'): The method of eigendecomposition to
%	  use. Choices are full eigendecomposition ('full') or Lanczos
%	  iteration ('lanczos'), which is recommended for sparse matrices.
%	- disc_type (default: 'kmeans'): The method of discretization to use.
%	  Choices are k-means ('kmeans') and Yu and Shi's (2003) method of
%	  orthogonal transformations ('ortho'). 'kmeans' requires the
%	  Statistics Toolbox, and 'ortho' requires Yu and Shi's (2003) NCut
%	  clustering toolbox.
%	- num_reps (default: 1): The number of times the k-means algorithm
%	  should be run to determine the final clustering. Has no effect when
%	  discretizing by orthogonal transformations.
%	- remove_cc (default: false): Set to true to remove all connected
%	  components aside from the giant connected component before performing
%	  eigendecomposition. The clustering result would then contain k + c
%	  clusters, where c is the number of connected components. It is
%	  recommended to set remove_cc to true if there are a lot of such
%	  components, because they will be identified much more quickly than by
%	  eigendecomposition. Requires the Bioinformatics Toolbox.
%	- output (default: 0): Set to 0 to suppress most output messages and to
%	  higher numbers to display progressively more output.
% 
% Additional outputs can be obtained by specifying them as follows: 
% [clu,W_bar,alpha_est,eigenvectors,eigenvalues] = batch_affect_spectral(...)
%	- W_bar: Length T cell array of smoothed similarity matrices.
%	- alpha_est: M-by-T matrix of estimated forgetting factor at each
%	  iteration. M is specified in the second cell of input parameter alpha.
%	- eigenvectors: Length T cell array of leading eigenvectors.
%	- eigenvalues: Length T cell array of leading eigenvalues.
% 
% Author: Kevin Xu

ip = inputParser;
ip.addRequired('ids',@iscell);
ip.addRequired('W',@iscell);
ip.addRequired('num_clust');
ip.addParamValue('max_clust',10,@(x)floor(x)==x);
ip.addParamValue('alpha','estimate');
ip.addParamValue('num_iter',3,@(x)floor(x)==x);
ip.addParamValue('initialize','previous');
ip.addParamValue('objective','NC');
ip.addParamValue('eig_type','lanczos');
ip.addParamValue('disc_type','kmeans');
ip.addParamValue('num_reps',1);
ip.addParamValue('remove_cc',false,@(x)islogical(x));
ip.addParamValue('output',0,@(x)(floor(x)==x) && (x>=0));
ip.parse(ids,W,num_clust,varargin{:});
max_clust = ip.Results.max_clust;
alpha = ip.Results.alpha;
num_iter = ip.Results.num_iter;
initialize = ip.Results.initialize;
objective = ip.Results.objective;
eig_type = ip.Results.eig_type;
disc_type = ip.Results.disc_type;
num_reps = ip.Results.num_reps;
remove_cc = ip.Results.remove_cc;
output = ip.Results.output;

t_max = length(W);
% Validate number of clusters
num_ev_plot = 10;	% Number of eigenvalues to plot, if necessary
if isnumeric(num_clust)
	% Scalar or vector specifying number of clusters at each time step
	if isscalar(num_clust)
		num_clust = num_clust*ones(1,t_max);
	end
	assert(length(num_clust) == t_max, ...
		'Length of num_clust must equal the number of time steps');
else
	% Use specified heuristic for choosing number of clusters
	if strcmp(num_clust,'eigengap_var')
		m = 0;
		num_ev_plot = max_clust;
	elseif strcmp(num_clust,'eigengap')
		num_ev_plot = max_clust;
	elseif strcmp(num_clust,'modularity') || strcmp(num_clust, ...
			'silhouette')
		m = 2:max_clust;
	else
		error(['num_clust must be one of ' ...
			'''eigengap'', ''modularity'', or ''silhouette'''])
	end
end

% Validate alpha and other parameters related to forgetting factor
if isnumeric(alpha)
	num_iter = 0;
	assert((alpha>=0) && (alpha<=1),'alpha must be between 0 and 1');
else
	if ~strcmp(alpha,'estimate')
		error('alpha must either be a number or ''estimate''');
	end
	if ~(strcmp(initialize,'previous') || strcmp(initialize,'ordinary'))
		error('initialize must either be ''previous'' or ''ordinary''')
	end
end

% Normalize eigenvectors before running k-means for normalized cut spectral
% clustering
if strcmp(objective,'NC')
	norm_ev = true;
else
	norm_ev = false;
end

% Initialize variable sizes
alpha_est = zeros(num_iter,t_max);
clu = cell(1,t_max);
W_bar = cell(1,t_max);
eigenvectors = cell(1,t_max);
eigenvalues = cell(1,t_max);

% For the eigengap heuristic compute the mean similarity matrix over all
% time steps and plot the leading eigenvalues
if strcmp(num_clust,'eigengap')
	num_clust = num_clusters_eigengap(ids,W,objective,num_ev_plot) ...
		* ones(1,t_max);
end

for t = 1:t_max
	if output > 0
		disp(['Processing time step ' int2str(t)])
	end
	
	if isnumeric(num_clust)
		m = num_clust(t);
	end
	
	n = length(ids{t});	
	% Only do temporal smoothing if not the first time step
	if t > 1
		% Identify rows and columns of new objects in current similarity
		% matrix
		[both_tf,both_loc] = ismember(ids{t},ids{t-1});
		W_prev = W_bar{t-1}(both_loc(both_tf),both_loc(both_tf));
		clu_prev = clu{t-1}(both_loc(both_tf));
		new_tf = ~both_tf;
		
		% Initialize smoothed similarity matrix
		if issparse(W{t})
			W_bar{t} = sparse(n,n);
		else
			W_bar{t} = zeros(n,n);
		end
		
		% No smoothing for new objects
		W_bar{t}(new_tf,:) = W{t}(new_tf,:);
		W_bar{t}(:,new_tf) = W{t}(:,new_tf);
		
		% Smoothing for objects present at both time steps
		if strcmp(alpha,'estimate')
			% Initialize current clustering result to be previous clustering
			% result or the result of one run of ordinary clustering
			clu{t} = zeros(n,1);
			if strcmp(initialize,'previous')
				clu{t}(both_tf) = clu_prev;
			else
				[clu{t},eigenvectors{t},eigenvalues{t}] = spectral_cluster ...
					(W{t},m,'objective',objective,'eig_type',eig_type, ...
					'disc_type',disc_type,'num_reps',num_reps,'norm_ev', ...
					norm_ev,'num_ev_plot',num_ev_plot,'remove_cc',remove_cc);
			end
			
			% Estimate alpha iteratively
			for iter = 1:num_iter
				alpha_est(iter,t) = estimate_alpha(W{t}(both_tf,both_tf), ...
					W_prev,clu{t}(both_tf));
				W_bar{t}(both_tf,both_tf) = alpha_est(iter,t)*W_prev ...
					+ (1-alpha_est(iter,t))*W{t}(both_tf,both_tf);
				[clu{t},eigenvectors{t},eigenvalues{t}] = spectral_cluster ...
					(W_bar{t},m,'objective',objective,'eig_type',eig_type, ...
					'disc_type',disc_type,'num_reps',num_reps,'norm_ev', ...
					norm_ev,'num_ev_plot',num_ev_plot,'remove_cc',remove_cc);
			end
		else
			W_bar{t}(both_tf,both_tf) = alpha*W_prev + (1-alpha) ...
				*W{t}(both_tf,both_tf);
			
			% Perform ordinary spectral clustering on W_bar
			[clu{t},eigenvectors{t},eigenvalues{t}] = spectral_cluster ...
				(W_bar{t},m,'objective',objective,'eig_type',eig_type, ...
				'disc_type',disc_type,'num_reps',num_reps,'norm_ev', ...
				norm_ev,'num_ev_plot',num_ev_plot,'remove_cc',remove_cc);			
		end
	else
		W_bar{t} = W{t};
		
		% Perform ordinary spectral clustering
		[clu{t},eigenvectors{t},eigenvalues{t}] = spectral_cluster ...
			(W_bar{t},m,'objective',objective,'eig_type',eig_type, ...
			'disc_type',disc_type,'num_reps',num_reps,'norm_ev', ...
			norm_ev,'num_ev_plot',num_ev_plot,'remove_cc',remove_cc);
	end
	
	% Select optimal number of clusters
	if ~isnumeric(num_clust)
		if strcmp(num_clust,'silhouette')
			[clu{t},avg_width] = select_clu_silhouette(W_bar{t}, ...
				clu{t},'similarity');
		elseif strcmp(num_clust,'modularity')
			[clu{t},Q] = select_clu_modularity(W_bar{t},clu{t},0);
		end
	end
	
	if t > 1	
		% Match clusters (using greedy method if more than 4 clusters)
		k = length(unique(clu{t}));
		if k > 4
			clu{t} = permute_clusters_greedy(ids{t},clu{t},ids{t-1}, ...
				clu{t-1},unmatched);
		else
			clu{t} = permute_clusters_opt(ids{t},clu{t},ids{t-1}, ...
				clu{t-1},unmatched);
		end
		unmatched = max(unmatched,max(clu{t})+1);
	else
		unmatched = max(clu{t})+1;
	end
end

function k = num_clusters_eigengap(ids,W,objective,num_ev_plot)

% Create mean similarity matrix over all time steps
W_seq = cell2matseq(ids,[],W,'union');
% W_seq will be a cell array if the similarity matrices are sparse and a
% 3-D matrix if the similarity matrices are full
if iscell(W_seq)
	t_max = length(W_seq);
	n = size(W_seq{1},1);
	W = sparse(n,n);
	for t = 1:t_max
		W = W + W_seq{t}/t_max;
	end
	
	if strcmp(objective,'AA')
		eigenvalues = eigs(W,num_ev_plot,'LA');
	else
		% Add eps to prevent zero entries, which cannot be inverted
		d = sum(abs(W),2) + eps;
		if strcmp(objective,'NC')
			D_inv_sqrt = spdiags(1./sqrt(d),0,n,n);
			L = speye(n) - D_inv_sqrt*W*D_inv_sqrt;
		else
			D = spdiags(d,0,n,n);
			L = D - W;
		end
		L = (L+L')/2;
		eigenvalues = eigs(L,num_ev_plot,'SA');
	end
else
	n = size(W_seq,1);
	W = mean(W_seq,3);
	
	if strcmp(objective,'AA')
		eigenvalues = sort(eig(W),'descend');
	else
		% Add eps to prevent zero entries, which cannot be inverted
		d = sum(abs(W),2) + eps;
		D = diag(d);
		if strcmp(objective,'NC')
			L = eye(n) - D^-0.5*W*D^-0.5;
		else
			L = D - W;
		end
		L = (L+L')/2;
		eigenvalues = sort(eig(L));
	end
end
plot(real(eigenvalues(1:min(n,num_ev_plot))),'x')
k = input('Enter the number of clusters: ');
