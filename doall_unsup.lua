dofile 'load_unsup.lua'
dofile 'model_unsup.lua'
dofile 'train.lua'
dofile 'valid.lua'
dofile 'test.lua'

while true do
   train()
   valid()
   if epoch % 2 == 0 then
     test()
   end
   collectgarbage()
end
