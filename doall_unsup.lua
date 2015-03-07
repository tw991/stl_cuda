dofile 'load_unsup.lua'
dofile 'model_unsup.lua'
dofile 'train.lua'
dofile 'test.lua'

while true do
   train()
   test()
   collectgarbage()
end
