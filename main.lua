require 'torch'
require 'nn'
require 'optim'


opt = {
   dataset = 'folder',      
   batchSize = 128,
   loadSize = 64,
   fineSize = 64,
   ngf = 96,               -- #  of gen filters in first conv layer
   ndf = 96,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
   noise = 'normal',       -- uniform / normal
   optimizer = 'sgd', 
   load_cp = 0,
}



-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('data.lua')

smn = torch.sum(mn)

----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local netG = nn.Sequential()

-- input is (nc) x 64 x 64
netG:add(SpatialConvolution(nc, ngf, 4, 4, 2, 2, 1, 1))
netG:add(nn.LeakyReLU(0.2, true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*8) x 4 x 4

-- netG:add(SpatialConvolution(ngf * 8, ngf * 16, 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ndf * 16)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*16) x 2 x 2

-- netG:add(SpatialFullConvolution(ngf*16, ngf * 8, 4, 4))
-- netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

netG:apply(weights_init)


local netD = nn.Sequential()

-- input is (nc) x 64 x 64
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)



local netA = nn.Sequential()

-- input is (nc*2) x 64 x 64
netA:add(SpatialConvolution(nc*2, ndf, 4, 4, 2, 2, 1, 1))
netA:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netA:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netA:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netA:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netA:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netA:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netA:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netA:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netA:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netA:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netA:apply(weights_init)

local criterion = nn.BCECriterion()

print('netG:',netG)
print('netA:',netA)
print('netD:',netD)

if opt.load_cp > 0 then
    epoch = opt.load_cp
    require 'cunn'
    require 'cudnn'
    netG = torch.load('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7')
    netD = torch.load('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7')
    netA = torch.load('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_A.t7')
end




local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

local input_img = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local ass_label = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noass_label = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize, 1)

if opt.gpu > 0 then
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    input_img = input_img:cuda()
    ass_label = ass_label:cuda()
    noass_label = noass_label:cuda()
    label = label:cuda()

    if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG, cudnn)
      cudnn.convert(netD, cudnn)
      cudnn.convert(netA, cudnn)
    end
    netD:cuda();           
    netG:cuda();           
    netA:cuda();   
    criterion:cuda();
end


local parametersD, gradParametersD = netD:getParameters()
local parametersA, gradParametersA = netA:getParameters()
local parametersG, gradParametersG = netG:getParameters()


local function load_data()
    data_tm:reset(); data_tm:resume()
    local batch = getbatch()
    input_img:copy(batch[{{},3}])
    ass_label:copy(batch[{{},1}])
    noass_label:copy(batch[{{},2}])
    data_tm:stop()
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()
   -- train with real   
   label:fill(real_label)
   local output = netD:forward(ass_label)
   
   local errD_real1 = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(ass_label, df_do)
   
   -- train with real (not associated)
   label:fill(real_label)
   local output = netD:forward(noass_label)
    
   local errD_real2 = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(noass_label, df_do)
    
   -- train with fake
   local fake = netG:forward(input_img)
   label:fill(fake_label)
   local output = netD:forward(fake)
    
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(fake, df_do)

   errD = (errD_real1 + errD_real2 + errD_fake)/3
   return errD, gradParametersD:mul(1/3)
end


-- create closure to evaluate f(X) and df/dX of domain discriminator
local fAx = function(x)
   gradParametersA:zero()
    
   local assd = torch.cat(input_img, ass_label, 2)
   local noassd = torch.cat(input_img, noass_label, 2)
   local fake = netG:forward(input_img)
   local faked = torch.cat(input_img, fake, 2)

   -- train with associated   
   label:fill(real_label)
   local output = netA:forward(assd)

   local errA_real1 = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netA:backward(assd, df_do)

   -- train with not associated
   label:fill(fake_label)
   local output = netA:forward(noassd)
   
   local errA_real2 = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netA:backward(noassd, df_do)

   -- train with fake
   label:fill(fake_label)
   local output = netA:forward(faked)
   
   local errA_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netA:backward(faked, df_do)

   errA = (errA_real1 + errA_real2 + errA_fake)/3
   return errA, gradParametersA:mul(1/3)
end


-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()
   --[[ the three lines below were already executed in fDx, so save computation
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   
   local fake = netG:forward(input_img)
   local output = netD:forward(fake)
   
   label:fill(real_label) -- fake labels are real for generator cost

   errGD = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput(fake, df_do)
   netG:backward(input_img, df_dg)
   
   local faked = torch.cat(input_img, fake, 2)
   local output = netA:forward(faked)
   label:fill(real_label) -- fake labels are real for generator cost
   errGA = criterion:forward(output, label)    
   local df_do = criterion:backward(output, label)
   local df_dg2 = netA:updateGradInput(faked, df_do)
   -- print(df_dg2:size())
   local df_dg = df_dg2[{{},{4,6}}]
   -- print(df_dg:size()) 
   netG:backward(input_img, df_dg)
   errG = (errGA + errGD)/2
   return errG, gradParametersG:mul(1/2)
end

if opt.display then disp = require 'display' end


if opt.optimizer == 'adam' 
    then optimizer = optim.adam
    else optimizer = optim.sgd
end

result = {}
local disp_config = {
  title = "error over time",
  labels = {"samples", "errD", "errG", "errA"},
  ylabel = "error",
  win=opt.display_id*2,
}

-- train
for epoch = opt.load_cp + 1, opt.load_cp + opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, smn, opt.batchSize do
      tm:reset()
    
      load_data()
        
        
      -- (0) Update D network
      optimizer(fDx, parametersD, optimStateD)
    
      -- (1) Update A network
      optimizer(fAx, parametersA, optimStateA)
      
      -- (2) Update G network
      optimizer(fGx, parametersG, optimStateG)
    
      


      -- display
      counter = counter + 1
      if counter % 20 == 0 and opt.display then
          local fake = netG:forward(input_img)
          local real = ass_label
          disp.image(torch.cat(fake,real,3):cat(input_img,3), {win=opt.display_id, title=opt.name})
      end
      
      -- logging
      if ((i-1) / opt.batchSize) % 10 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f Err_A: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(smn / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1, errA and errA or -1))
          
          table.insert(result, {i + smn*(epoch-1), errD, errG, errA})
          disp.plot(result, disp_config)
          
            
      end
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersA, gradParametersA = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_A.t7', netA:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersA, gradParametersA = netA:getParameters()
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end