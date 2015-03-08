require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function valid()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on valid set:')
   for t = 1,valsize,batchSize do
      -- disp progress
      xlua.progress(t, valsize)

      -- get new sample
      local input = valdata.X[{{t, math.min(t+batchSize-1, testsize)}}]:cuda()
      local target = valdata.y[{{t, math.min(t+batchSize-1, testsize)}}]:cuda()

      -- test sample
      local pred = model:forward(input)
      -- print("\n" .. target .. "\n")
      confusion:batchAdd(pred, target)
      collectgarbage()
   end

   -- timing
   time = sys.clock() - time
   time = time / testsize
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end

   -- next iteration:
   confusion:zero()
end
