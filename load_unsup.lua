require 'cunn'
require 'nn'
require 'cutorch'
require 'image'
require 'xlua'
require 'unsup'
matio = require 'matio'

torch.setdefaulttensortype('torch.CudaTensor')
print("==> load dataset")
traindataload = matio.load('/scratch/courses/DSGA1008/A2/matlab/train.mat')
traindataload.X = torch.reshape(traindataload.X, 5000,3,96,96):float()
traindataload.X = traindataload.X:transpose(3,4)
testdata = matio.load('/scratch/courses/DSGA1008/A2/matlab/test.mat')
testdata.X = torch.reshape(testdata.X, 8000,3,96,96):float()
testdata.X = testdata.X:transpose(3,4)
--data_fd = torch.DiskFile("/scratch/courses/DSGA1008/A2/binary/unlabeled_X.bin", "r", true)
--data_fd:binary():littleEndianEncoding()
--unlabeledtotalsize = 100000
--unlabeled = torch.ByteTensor(unlabeledtotalsize, 3, 96, 96)
--data_fd:readByte(unlabeled:storage())
--unlabeled= unlabeled:float()
--data_fd = nil

-- set parameters
patchSize = 7
ncenter = 64
trsize = 4500
valsize = 500
testsize = 8000
--unlabelsize = 100000
traindata = {}
traindata.X = torch.Tensor(trsize, 3,96,96)
traindata.y = torch.Tensor(trsize, 1)
valdata = {}
valdata.X = torch.Tensor(valsize, 3, 96, 96)
valdata.y = torch.Tensor(valsize, 1)

-- Train Valid Split
print("==> Split training/validation sets")
index_train = torch.randperm(5000)
for i =1,trsize do
	traindata.X[i] = traindataload.X[index_train[i]]
	traindata.y[i] = traindataload.y[index_train[i]]
	collectgarbage()
end
for i =1,valsize do
	valdata.X[i] = traindataload.X[index_train[i+trsize]]
	valdata.y[i] = traindataload.y[index_train[i+trsize]]
	collectgarbage()
end
traindataload = nil

-- Normalization

print("==> Normalization")
for i = 1,trsize do
        xlua.progress(i, trsize)
        traindata.X[i] = traindata.X[i]:div(torch.max(torch.abs(traindata.X[i])))
        traindata.X[i] = traindata.X[i]:add(torch.mean(traindata.X[i]))
	collectgarbage()
end

print("==> for test")
for i = 1,testsize do
        xlua.progress(i, testsize)
        testdata.X[i] = testdata.X[i]:div(torch.max(torch.abs(testdata.X[i])))
        testdata.X[i] = testdata.X[i]:add(torch.mean(testdata.X[i]))
	collectgarbage()
end
collectgarbage()
