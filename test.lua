require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()
   local prediction = torch.zeros(testsize)

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testsize,batchSize do
      -- disp progress
      xlua.progress(t, testsize)

      -- get new sample
      local input = testdata.X[{{t, math.min(t+batchSize-1, testsize)}}]:cuda()
      local target = testdata.y[{{t, math.min(t+batchSize-1, testsize)}}]:cuda()

      -- test sample
      local pred = model:forward(input:cuda())
      -- print("\n" .. target .. "\n")
      confusion:batchAdd(pred:float(), target)
      local val, loc = torch.max(pred,2)
      prediction[{{t, math.min(t+batchSize-1, testsize)}}] = loc
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

   -- output prediction
   file = io.open("prediction.csv", "w")
   file:write("Id,Prediction\n")
   for i =1,testsize do
      file:write(tostring(i)..","..tostring(prediction[i]).."\n")
   end
   file:flush()
   file:close()
   -- next iteration:
   confusion:zero()
end
