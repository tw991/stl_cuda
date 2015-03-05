require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
model:cuda()   
criterion:cuda()
batchSize = 128

print '==> defining some tools'

-- classes
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat('results', 'train.log'))
testLogger = optim.Logger(paths.concat('results', 'test.log'))


----------------------------------------------------------------------
print '==> configuring optimizer'

   optimState = {
      learningRate = 1e-3,
      weightDecay = 0,
      momentum = 0,
      learningRateDecay = 0.5
   }
   optimMethod = optim.sgd

----------------------------------------------------------------------

print '==> defining training procedure'

function train()
	epoch = epoch or 1
	local time = sys.clock()
	local parameters,gradParameters = model:getParameters()
	print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
	local clr = 0.1
	for t = 1,trsize,batchSize do
		xlua.progress(t, trsize)
		local inputs = traindata.X[{{t, math.min(t+batchSize-1, trsize)}}]:cuda()
		local targets = traindata.y[{{t, math.min(t+batchSize-1, trsize)}}]:cuda()
		gradParameters:zero()
		local output = model:forward(inputs)
		local f = criterion:forward(output, targets)
		local trash, argmax = output:max(2)
		confusion:batchAdd(output, targets)
		--no_wrong = no_wrong + torch.ne(argmax, targets):sum()
		model:backward(inputs, criterion:backward(output, targets))
		clr = optimState.learningRate * (0.5^math.floor(epoch/optimState.learningRateDecay))
		parameters:add(-clr, gradParameters)
		collectgarbage() 
	end
	time = sys.clock() - time
	time = time / trsize
	print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
	print(confusion)
	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	local filename = paths.concat('results', 'model.net')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	print('==> saving model to '..filename)
	-- torch.save(filename, model)
	confusion:zero()
	epoch = epoch + 1
end
