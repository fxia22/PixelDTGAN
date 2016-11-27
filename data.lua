require 'image'

loadSize = {64,64}
function loadImage(path)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   local iW = input:size(3)
   local iH = input:size(2)
   if iW < iH then
      input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   else
      input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   end
   return input:mul(2):csub(1)
end


cloth_table = torch.load('cloth_table.t7')
models_table = torch.load('models_table.t7')

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end


cn = tablelength(cloth_table)
mn = torch.Tensor(cn)

for k,v in pairs(models_table) do
    mn[k] = tablelength(v)
end



function getbatch()
    batch = torch.Tensor(128,3,3,64,64)
    for i = 1,128 do
        seed = torch.random(1, 100000) -- fix seed
        gen = torch.Generator()
        torch.manualSeed(gen, i*seed)
        r1 = torch.random(gen,1,cn)
        r2 = torch.random(gen,1,cn)
        r3 = torch.random(gen,1,mn[r1])

        path1 = cloth_table[r1]
        path2 = cloth_table[r2]
        path3 = models_table[r1][r3]

        img1 = loadImage(path1)
        img2 = loadImage(path2)
        img3 = loadImage(path3)
        
        batch[i][1] = img1
        batch[i][2] = img2
        batch[i][3] = img3
    end
    return batch
end

