nstate = {512,512,512,128}
filtersize = 9
poolsize = 2
normkernel = image.gaussian1D(9)
noutputs = 10

model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(3, nstate[1], filtersize, filtersize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

model:add(nn.SpatialConvolutionMM(nstate[1], nstate[2], filtersize, filtersize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

model:add(nn.SpatialConvolutionMM(nstate[2], nstate[3], filtersize, filtersize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

model:add(nn.View(nstate[3]*5*5))
model:add(nn.Linear(nstate[3]*5*5, nstate[4]))
model:add(nn.ReLU())
model:add(nn.Linear(nstate[4], noutputs))

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
