require 'cunn'
require 'nn'
require 'cutorch'
require 'image'
require 'xlua'
require 'unsup'
matio = require 'matio'

-- torch.setdefaulttensortype('torch.CudaTensor')
print("==> load dataset")
traindata = matio.load('/scratch/courses/DSGA1008/A2/matlab/train.mat')
traindata.X = torch.reshape(traindata.X, 5000,3,96,96):float()
traindata.X = traindata.X:transpose(3,4)
testdata = matio.load('/scratch/courses/DSGA1008/A2/matlab/test.mat')
testdata.X = torch.reshape(testdata.X, 8000,3,96,96):float()
testdata.X = testdata.X:transpose(3,4)

-- set parameters
patchSize = 16
numPatches = 10000
ncenter = 128	
trsize = 5000
testsize = 8000


-- randomly pick patches from unlabeled data
print("==> extract patches")
patches = torch.zeros(numPatches, 3, patchSize, patchSize):float()
for i = 1,numPatches do
	xlua.progress(i, numPatches) 
	-- Show Progress
	local patch_x = torch.random(traindata.X:size(3)-patchSize)
	local patch_y = torch.random(traindata.X:size(4)-patchSize)
	local patch_id = torch.random(traindata.X:size(1))
	-- Random Pick Patch
	local random_patch = traindata.X[{patch_id,{},{patch_x,patch_x+patchSize-1},{patch_y,patch_y+patchSize-1}}]
	patches[i] = random_patch
	-- Locally Normalize Patches
	-- Divide by max(abs)
	-- Subtract by mean
	if torch.abs(torch.max(patches[i])) ~= 0 then
		patches[i] = patches[i]:div(torch.max(torch.abs(patches[i])))
		patches[i] = patches[i]:add(torch.mean(patches[i]))
	end
end


-- ZCA-Whitening
patches = unsup.zca_whiten(patches:double()):float()

-- doing k-means feature extraction
--
-- unsup.kmean() only works for double tensor
print("==> using K-Means to find filter")
kernels, count = unsup.kmeans(patches:double(), ncenter, 10)
kernels = kernels:float()
count = nil

-- forward through a layer of CNN with weight = kernels
print("==> Use K-means centeroid to do convolution")
firstnet = nn.SpatialConvolutionMM(3,ncenter,patchSize,patchSize)
firstnet:cuda()
firstnet.weight = kernels:cuda()
firstnet.bias:zero()
secondnet = nn.ReLU()
secondnet:cuda()
firstout = torch.Tensor(trsize, ncenter, 96-patchSize+1, 96-patchSize+1)
cuda_batch = 200
for i = 1,trsize, cuda_batch do
	xlua.progress(i,trsize)
	firstout[{{i,i+cuda_batch-1},{},{},{}}]=secondnet:forward(firstnet:forward(traindata.X[{{i, i+cuda_batch-1},{},{},{}}]:cuda())):float()
end
traindata.X = nil
traindata.X = firstout
firstout = nil
collectgarbage()
firstout = torch.Tensor(testsize, ncenter, 96-patchSize+1, 96-patchSize+1)
for i =1,testsize, cuda_batch do
	xlua.progress(i,testsize)
	firstout[{{i,i+cuda_batch-1},{},{},{}}]=secondnet:forward(firstnet:forward(testdata.X[{{i, i+cuda_batch-1},{},{},{}}]:cuda())):float()
end
testdata.X = nil
testdata.X = firstout
firstnet = nil
secondnet = nil
firstout = nil
collectgarbage()
