dofile 'load.lua'
dofile 'model.lua'
dofile 'train.lua'
dofile 'test.lua'

while true do
   train()
   test()
   collectgarbage()
end
